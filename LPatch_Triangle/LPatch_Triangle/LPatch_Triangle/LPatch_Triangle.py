import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import fnmatch
import  numpy as np
import random
import math




#
# CreateSemiLMPatches
#

class LPatch_Triangle(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "LPatch_Triangle" # TODO make this more human readable by adding spaces
    self.parent.categories = ["LTools"]
    self.parent.dependencies = []
    self.parent.contributors = ["Michael Severinsen (CCEM)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
      This module interactively places patches of semi-landmarks between user-specified anatomical landmarks.
      <p>For more information see the <a href="https://github.com/SlicerMorph/SlicerMorph/tree/master/Docs/CreateSemiLMPatches">online documentation.</a>.</p>
      """
    #self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
      This module is developed by Michael Lind Severinsen, supported by (VIL40582) from VILLUM FONDEN, as part of my PhD Thesis
      This module is based on the work from by Sara Rolfe, and Murat Maga for SlicerMorph. SlicerMorph was originally supported by an NSF/DBI grant, "An Integrated Platform for Retrieval, Visualization and Analysis of 3D Morphology From Digital Biological Collections"
      awarded to Murat Maga (1759883), Adam Summers (1759637), and Douglas Boyer (1759839).
      https://nsf.gov/awardsearch/showAward?AWD_ID=1759883&HistoricalAwards=false
      """ # replace with organization, grant and thanks.

    # Additional initialization step after application startup is complete
    #slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


#
# CreateSemiLMPatchesWidget
#

class LPatch_TriangleWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  def onMeshSelect(self):
    self.applyButton.enabled = bool (self.meshSelect.currentNode() and self.LMSelect.currentNode())
    nodes=self.fiducialView.selectedIndexes()
    self.mergeButton.enabled = bool (nodes and self.LMSelect.currentNode() and self.meshSelect.currentNode())

  def onLMSelect(self):
    self.applyButton.enabled = bool (self.meshSelect.currentNode() and self.LMSelect.currentNode())
    nodes=self.fiducialView.selectedIndexes()
    self.mergeButton.enabled = bool (nodes and self.LMSelect.currentNode() and self.meshSelect.currentNode())

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    # 3D view set up tab
    self.meshSelect = slicer.qMRMLNodeComboBox()
    self.meshSelect.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.meshSelect.selectNodeUponCreation = False
    self.meshSelect.addEnabled = False
    self.meshSelect.removeEnabled = False
    self.meshSelect.noneEnabled = True
    self.meshSelect.showHidden = False
    self.meshSelect.setMRMLScene( slicer.mrmlScene )
    self.meshSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onMeshSelect)
    parametersFormLayout.addRow("Model: ", self.meshSelect)
    self.meshSelect.setToolTip( "Select model node for semilandmarking" )

    self.LMSelect = slicer.qMRMLNodeComboBox()
    self.LMSelect.nodeTypes = ( ('vtkMRMLMarkupsFiducialNode'), "" )
    self.LMSelect.selectNodeUponCreation = False
    self.LMSelect.addEnabled = False
    self.LMSelect.removeEnabled = False
    self.LMSelect.noneEnabled = True
    self.LMSelect.showHidden = False
    self.LMSelect.showChildNodeTypes = False
    self.LMSelect.setMRMLScene( slicer.mrmlScene )
    self.LMSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onLMSelect)
    parametersFormLayout.addRow("Landmark set: ", self.LMSelect)
    self.LMSelect.setToolTip( "Select the landmark set that corresponds to the model" )

     #
    # input landmark numbers for grid
    #

    gridPointsLayout= qt.QGridLayout()
    self.landmarkGridPoint1 = ctk.ctkDoubleSpinBox()
    self.landmarkGridPoint1.minimum = 0
    self.landmarkGridPoint1.singleStep = 1
    self.landmarkGridPoint1.setDecimals(0)
    self.landmarkGridPoint1.value = 1 # PREFIXED; SHOULD BE DELETED 
    self.landmarkGridPoint1.setToolTip("Input the landmark numbers to define center - this will have the shared corner")

    self.landmarkGridPoint2 = ctk.ctkDoubleSpinBox()
    self.landmarkGridPoint2.minimum = 0
    self.landmarkGridPoint2.singleStep = 1
    self.landmarkGridPoint2.setDecimals(0)
    self.landmarkGridPoint2.value = 2 # PREFIXED; SHOULD BE DELETED 
    self.landmarkGridPoint2.setToolTip("Input the landmark numbers to create vector from landmark 1 to this")

    self.landmarkGridPoint3 = ctk.ctkDoubleSpinBox()
    self.landmarkGridPoint3.minimum = 0
    self.landmarkGridPoint3.singleStep = 1
    self.landmarkGridPoint3.setDecimals(0)
    self.landmarkGridPoint3.value = 4 # PREFIZED; SHOULD BE DELETED 
    self.landmarkGridPoint3.setToolTip("Input the landmark numbers to create vector from landmark 2 to this")

    gridPointsLayout.addWidget(self.landmarkGridPoint1,1,2)
    gridPointsLayout.addWidget(self.landmarkGridPoint2,1,3)
    gridPointsLayout.addWidget(self.landmarkGridPoint3,1,4)

    parametersFormLayout.addRow("Semi-landmark grid points:", gridPointsLayout)
    #
    # input landmark numbers for grid
        #
    # INSERT_YOUR_CODE

    # Main checkbox: Use point-based outlines (default is UNCHECKED = use fixed outlines)
    self.usePointBasedOutlinesCheckBox = qt.QCheckBox("Use point-based outlines")
    self.usePointBasedOutlinesCheckBox.setChecked(False)  # Default: use fixed outlines
    self.usePointBasedOutlinesCheckBox.setToolTip("If checked, define outlines using specific landmark points. If unchecked, use fixed grid sampling.")
    parametersFormLayout.addRow(self.usePointBasedOutlinesCheckBox)

    #
    # DEFAULT MODE: Fixed outlines with grid sampling (visible by default)
    #
    self.gridSamplingRate = ctk.ctkDoubleSpinBox()
    self.gridSamplingRate.minimum = 3
    self.gridSamplingRate.maximum = 50
    self.gridSamplingRate.singleStep = 1
    self.gridSamplingRate.setDecimals(0)
    self.gridSamplingRate.value = 10
    self.gridSamplingRate.setToolTip("Number of rows/columns in the triangular grid")
    parametersFormLayout.addRow("Grid sample rate:", self.gridSamplingRate)

    #
    # POINT-BASED MODE: Manual outline definition (hidden by default)
    #



    # Manual outline inputs (3 text fields for user input)
    gridPointsOutline = qt.QGridLayout()

    self.outlinePointsInput1 = qt.QLineEdit()
    self.outlinePointsInput1.setPlaceholderText("1. Enter comma-separated landmark numbers")
    self.outlinePointsInput1.setToolTip("Input comma-separated landmark numbers for first triangle line (e.g. 1,2,3,4,5)")
    self.outlinePointsInput1.setValidator(qt.QRegExpValidator(qt.QRegExp("[0-9,]*")))
    self.outlinePointsInput1.setText("1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,2")
    self.outlinePointsInput1.setToolTip("Landmark numbers,it has to started with the shared corner between line one and two")


    self.outlinePointsInput2 = qt.QLineEdit()
    self.outlinePointsInput2.setPlaceholderText("2. Enter comma-separated landmark numbers")
    self.outlinePointsInput2.setToolTip("Input comma-separated landmark numbers for second triangle line (e.g. 1,2,3,4,5)")
    self.outlinePointsInput2.setValidator(qt.QRegExpValidator(qt.QRegExp("[0-9,]*")))
    self.outlinePointsInput2.setText("1,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,4")
    self.outlinePointsInput2.setToolTip("Landmark numbers,it has to started with the shared corner between line one and two")

    # Unused atm, 
    #self.outlinePointsInput3 = qt.QLineEdit()
    #self.outlinePointsInput3.setPlaceholderText("3. Enter comma-separated landmark numbers")
    #self.outlinePointsInput3.setToolTip("Input comma-separated landmark numbers for third triangle line (e.g. 1,2,3,4,5)")
    #self.outlinePointsInput3.setValidator(qt.QRegExpValidator(qt.QRegExp("[0-9,]*")))
    #self.outlinePointsInput3.setText("4,2")

    # Add the three input fields to the layout
    gridPointsOutline.addWidget(self.outlinePointsInput1, 0, 0)
    gridPointsOutline.addWidget(self.outlinePointsInput2, 1, 0)
    #gridPointsOutline.addWidget(self.outlinePointsInput3, 2, 0)

    self.manualOutlineInputsWidget = qt.QWidget()
    self.manualOutlineInputsWidget.setLayout(gridPointsOutline)
    self.manualOutlineInputsWidget.setVisible(False)  # Initially hidden
    parametersFormLayout.addRow("Manual triangle line inputs:", self.manualOutlineInputsWidget)




    # Sub-checkbox: Use fixed number per triangle line
    self.useFixedNumberPerLineCheckBox = qt.QCheckBox("Use fixed number of landmarks per triangle line")
    self.useFixedNumberPerLineCheckBox.setChecked(True)  # Default when point-based is activated
    self.useFixedNumberPerLineCheckBox.setToolTip("If checked, uses same fixed number for all triangle lines. If unchecked, manually input landmark sequences.")
    self.useFixedNumberPerLineCheckBox.setVisible(False)  # Initially hidden
    parametersFormLayout.addRow(self.useFixedNumberPerLineCheckBox)

    # Fixed number input for triangle lines
    self.fixedNumberPerLine = qt.QSpinBox()
    self.fixedNumberPerLine.minimum = 3
    self.fixedNumberPerLine.maximum = 50
    self.fixedNumberPerLine.value = 10
    self.fixedNumberPerLine.setToolTip("Fixed number of landmarks for each triangle line")
    self.fixedNumberPerLine.setVisible(False)  # Initially hidden
    parametersFormLayout.addRow("Fixed number of landmarks per line:", self.fixedNumberPerLine)


    self.outLinesInterpolation = qt.QLineEdit()
    self.outLinesInterpolation.setPlaceholderText("2. Enter comma-separated numbers, to define number of landmarks for filling the triangle")
    self.outLinesInterpolation.setToolTip("Input comma-separated numbers for landmark to fill the triangle ")
    self.outLinesInterpolation.setValidator(qt.QRegExpValidator(qt.QRegExp("[0-9,]*")))
    self.outLinesInterpolation.setText("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27")
    self.outLinesInterpolation.setVisible(False)  # Initially hidden
    parametersFormLayout.addRow("Variable Number of landmarks per line:", self.outLinesInterpolation)

    # Function to update UI visibility based on user choices
    def updateUIVisibility():
        usePointBased = self.usePointBasedOutlinesCheckBox.isChecked()
        
        if usePointBased:
            # POINT-BASED MODE: Show point-based controls, hide grid controls
            self.useFixedNumberPerLineCheckBox.setVisible(True)
            self.fixedNumberPerLine.setVisible(True)

            self.outLinesInterpolation.setVisible(True)

            
            useFixedNumber = self.useFixedNumberPerLineCheckBox.isChecked()
            self.manualOutlineInputsWidget.setVisible(True)
            if useFixedNumber:
                # Show fixed number input, hide manual inputs
                self.fixedNumberPerLine.setEnabled(True)
                self.outLinesInterpolation.setEnabled(False)
            else:
                # Hide fixed number input, show manual inputs
                self.fixedNumberPerLine.setEnabled(False)
                self.outLinesInterpolation.setEnabled(True)
            
            # Hide grid sampling rate
            self.gridSamplingRate.setEnabled(False)
            
        else:
            # DEFAULT MODE: Fixed outlines with grid sampling
            # Hide all point-based controls
            self.useFixedNumberPerLineCheckBox.setVisible(False)
            self.fixedNumberPerLine.setVisible(False)
            self.manualOutlineInputsWidget.setVisible(False)
            self.outLinesInterpolation.setVisible(False)
            # Show and enable grid sampling rate
            self.gridSamplingRate.setEnabled(True)

    # Connect checkbox state changes to the update function
    self.usePointBasedOutlinesCheckBox.stateChanged.connect(updateUIVisibility)
    self.useFixedNumberPerLineCheckBox.stateChanged.connect(updateUIVisibility)

    # Set initial UI state (default: fixed outlines with grid sampling)
    updateUIVisibility()
    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Advanced menu
    #
    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = "Advanced"
    advancedCollapsibleButton.collapsed = True
    parametersFormLayout.addRow(advancedCollapsibleButton)

    # Layout within the dummy collapsible button
    advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)

    #
    # Maximum projection slider
    #
    self.projectionDistanceSlider = ctk.ctkSliderWidget()
    self.projectionDistanceSlider.singleStep = 1
    self.projectionDistanceSlider.minimum = 0
    self.projectionDistanceSlider.maximum = 100
    self.projectionDistanceSlider.value = 40
    self.projectionDistanceSlider.setToolTip("Set maximum projection distance as a percentage of image size")
    advancedFormLayout.addRow("Set maximum projection distance: ", self.projectionDistanceSlider)

    #
    # Normal smoothing slider
    #
    self.smoothingSlider = ctk.ctkSliderWidget()
    self.smoothingSlider.singleStep = 1
    self.smoothingSlider.minimum = 0
    self.smoothingSlider.maximum = 100
    self.smoothingSlider.value = 0
    self.smoothingSlider.setToolTip("Set smothing of normal vectors for projection")
    advancedFormLayout.addRow("Set smoothing of projection vectors: ", self.smoothingSlider)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Generate semilandmarks."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    #
    # Fiducials view
    #
    self.fiducialView = slicer.qMRMLSubjectHierarchyTreeView()
    self.fiducialView.setMRMLScene(slicer.mrmlScene)
    self.fiducialView.setMultiSelection(True)
    self.fiducialView.setAlternatingRowColors(True)
    self.fiducialView.setDragDropMode(True)
    self.fiducialView.setColumnHidden(self.fiducialView.model().transformColumn, True);
    self.fiducialView.sortFilterProxyModel().setNodeTypes(["vtkMRMLMarkupsFiducialNode"])
    parametersFormLayout.addRow(self.fiducialView)

    #
    # Apply Button
    #
    self.mergeButton = qt.QPushButton("Merge highlighted nodes")
    self.mergeButton.toolTip = "Generate a single merged landmark file from the selected nodes"
    self.mergeButton.enabled = False
    parametersFormLayout.addRow(self.mergeButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.mergeButton.connect('clicked(bool)', self.onMergeButton)
    self.fiducialView.connect('currentItemChanged(vtkIdType)', self.updateMergeButton)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass

  def onApplyButton(self):
      logic = CreateSemiLMPatchesLogic()
      enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
      gridLandmarks = [int(self.landmarkGridPoint1.value), int(self.landmarkGridPoint2.value), int(self.landmarkGridPoint3.value)]
      smoothingIterations = int(self.smoothingSlider.value)
      projectionRayTolerance = self.projectionDistanceSlider.value
      
      # Determine which mode we're in and prepare outLines accordingly
      usePointBased = self.usePointBasedOutlinesCheckBox.isChecked()
      
      if not usePointBased:
          # MODE 1: Grid sampling mode (default)
          # Generate triangle with gridSamplingRate points on each side
          sampleRate = int(self.gridSamplingRate.value)
          outLines = self.generateTriangleOutlines(gridLandmarks, sampleRate)

          outLines_interpolator = []
          for k in range(sampleRate, -1, -1):
              outLines_interpolator.append(k)
          outLines_interpolator.reverse()  # Reverse in place
      else:
          # MODE 2: Point-based outlines mode
          useFixedNumber = self.useFixedNumberPerLineCheckBox.isChecked()
          outLines = self.parseManualOutlinesRaw()
          if len(outLines[0]) != len(outLines[1]):
              print("length of outlines: " ,len(outLines[0]), ", " , len(outLines[1]))
              print("Must have the same lenght")
              sys.exit()
          
          if useFixedNumber:
              # MODE 2A: Fixed number per line
              outLines_interpolator = [int(self.fixedNumberPerLine.value)] * len(outLines[0])

   
              
          else:
              # MODE 2B: Manual outline 
              outLines_interpolatortxt = self.outLinesInterpolation.text.strip()
              outLines_interpolator = [int(x.strip()) for x in outLines_interpolatortxt.split(",") if x.strip()]
              
              # Double check lengths
              if len(outLines[0]) != len(outLines_interpolator):
                 print("length of outlines: " ,len(outLines[0]))
                 print("length of interpolation lines: " ,len(outLines_interpolator))
                 print("NOT MATCHING LENGTHS")
                 sys.exit()
      
      # All modes now have standardized outLines format
      # Call the logic with standardized parameters
      logic.run(
          self.meshSelect.currentNode(), 
          self.LMSelect.currentNode(), 
          gridLandmarks, 
          outLines, 
          int(self.gridSamplingRate.value)+1,  # sampleRate parameter maintained for compatibility
          smoothingIterations, 
          outLines_interpolator,
          projectionRayTolerance
      )

  def generateTriangleOutlines(self, gridLandmarks, sampleRate):
      """
      Generate triangle outlines with sampleRate points on each side.
      All outlines will have the same length (sampleRate).
      Returns coordinates, not indices.
      """
      LMNode = self.LMSelect.currentNode()
      
      def interpolate_line_with_coordinates(start_idx, end_idx, num_points):
          """Generate interpolated coordinates between two landmark points"""
          # Get actual 3D coordinates
          start_point = np.array(LMNode.GetNthControlPointPosition(int(start_idx-1)))
          end_point = np.array(LMNode.GetNthControlPointPosition(int(end_idx-1)))
          
          coordinates = []
          
          # Generate interpolated points
          for i in range(num_points):
              t = i / (num_points - 1) if num_points > 1 else 0
              interpolated_point = start_point + t * (end_point - start_point)
              coordinates.append(interpolated_point.tolist())
          
          return coordinates
      
      # Generate three lines of the triangle - ALL with same length (sampleRate)
      line1 = interpolate_line_with_coordinates(gridLandmarks[0], gridLandmarks[1], sampleRate)
      line2 = interpolate_line_with_coordinates(gridLandmarks[0], gridLandmarks[2], sampleRate) 
      #line3 = interpolate_line_with_coordinates(gridLandmarks[1], gridLandmarks[2], sampleRate)
      
      return [line1, line2]#, line3]


  def parseManualOutlinesRaw(self):
      """
      Parse the manual outline inputs from the text fields and convert indices to 3D coordinates.
      """
      LMNode = self.LMSelect.currentNode()
      outLines = []
      
      for outlineInput in [self.outlinePointsInput1, self.outlinePointsInput2]:#, self.outlinePointsInput3]:
          lmoutlinetext = outlineInput.text.strip()
          if lmoutlinetext:
              lm_indices = [int(x.strip()) for x in lmoutlinetext.split(",") if x.strip()]
              # Convert indices to 3D coordinates
              coordinates = []
              for idx in lm_indices:
                  point = LMNode.GetNthControlPointPosition(int(idx-1))  # Convert to 0-based index
                  coordinates.append([point[0], point[1], point[2]])
              outLines.append(coordinates)

          else:
              outLines.append([])  # Empty list for empty input
      
      return outLines


  def onMergeButton(self):
    logic = CreateSemiLMPatchesLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    logic.mergeTree(self.fiducialView, self.LMSelect.currentNode(), self.meshSelect.currentNode(),int(self.gridSamplingRate.value))

  def updateMergeButton(self):
    nodes=self.fiducialView.selectedIndexes()
    self.mergeButton.enabled = bool (nodes and self.LMSelect.currentNode() and self.meshSelect.currentNode())

#
# CreateSemiLMPatchesLogic
#

class CreateSemiLMPatchesLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  def run(self, meshNode, LMNode, gridLandmarks,outLines ,sampleRate, smoothingIterations,outLines_interpolator, maximumProjectionDistance=.25):

    if(smoothingIterations == 0):
      surfacePolydata = meshNode.GetPolyData()
      normalArray = surfacePolydata.GetPointData().GetArray("Normals")
      if(not normalArray):
        normalFilter=vtk.vtkPolyDataNormals()
        normalFilter.ComputePointNormalsOn()
        normalFilter.SetInputData(surfacePolydata)
        normalFilter.Update()
        normalArray = normalFilter.GetOutput().GetPointData().GetArray("Normals")
        if(not normalArray):
          print("Error: no normal array")
    else:
      print('smoothing normals')
      normalArray = self.getSmoothNormals(meshNode,smoothingIterations)
    semiLandmarks = self.applyPatch(meshNode, LMNode, gridLandmarks,outLines, sampleRate, normalArray, outLines_interpolator, maximumProjectionDistance)


    return True




  def applyPatch(self, meshNode, LMNode, gridLandmarks, outLines,sampleRate, polydataNormalArray, outLines_interpolator, maximumProjectionDistance=.25):
    print(" \n\n\n Start patcher \n\n\n")

    def calculate_ray_endpoint(p0, p1, modelPoint, ray_length=10):
        # Calculate the ray endpoint, by defining a plane by the cross product wit the two vectors: (modelpoint,p0) and (modelpoint,p1)

        # Calculate vectors from modelPoint to p0 and p1
        v1 = [p0[0] - modelPoint[0], p0[1] - modelPoint[1], p0[2] - modelPoint[2]]
        v2 = [p1[0] - modelPoint[0], p1[1] - modelPoint[1], p1[2] - modelPoint[2]]
        
        # Calculate cross product to get normal direction
        rayDirection = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ]
        
        # Calculate magnitude of rayDirection
        magnitude = (rayDirection[0]**2 + rayDirection[1]**2 + rayDirection[2]**2)**0.5
        #print("My ray driection is: ", rayDirection)
        # Normalize by dividing each component by magnitude
        if magnitude > 0:
            rayDirection = [d/magnitude for d in rayDirection]
        
        # Calculate both ray end points (positive and negative directions)
        rayEndPoint1 = [modelPoint[dim] + rayDirection[dim] * ray_length for dim in range(3)]
        rayEndPoint2 = [modelPoint[dim] - rayDirection[dim] * ray_length for dim in range(3)]
        
        return rayEndPoint1, rayEndPoint2
    


    # Project everything now, to make the new lies into the files 
    sourcePoints = vtk.vtkPoints()
    targetPoints = vtk.vtkPoints()

 
    semiPoints = vtk.vtkPoints()

    
    # Add first and last points from the first outline to semiPoints for compatibility
    if outLines and len(outLines[0]) >= 2:
        semiPoints.InsertNextPoint(outLines[0][0])   # First point of first outline
        semiPoints.InsertNextPoint(outLines[0][-1])  # Last point of first outline

    semiPolydata=vtk.vtkPolyData()
    semiPolydata.SetPoints(semiPoints)
    
    # Add outline endpoints to the transformation points
    for outline in outLines:
            # Add first and last point of each outline
            targetPoints.InsertNextPoint(outline[0])
            sourcePoints.InsertNextPoint(outline[0])


    ####################################################
    # Start triangulation part 
    ####################################################
    print("Processing filling the triangle")
    
    # Generate triangle fill points between the outlines
    # This creates the internal grid structure for the triangle
    num_pointers = []
    point_counter = 0
    
    # Fill the triangle with points, by going between the two shared edges. 
    for start_point,end_point,num_point in zip( outLines[0] , outLines[1],outLines_interpolator):

        print(num_point)
        range_start = point_counter
        start_point = np.array(start_point)
        end_point = np.array(end_point)
        for j in range(num_point):
            t = (j + 1) / (num_point + 1)  # Avoid endpoints
            internal_point = start_point + t * (end_point - start_point)
            semiPoints.InsertNextPoint(internal_point)
            point_counter += 1
        
        range_end = point_counter
        num_pointers.append([range_start, range_end])

    print("added: ", point_counter)
    # Add grid landmarks to transformation points (only once, not duplicated)
    for gridVertex in gridLandmarks:
        point = LMNode.GetNthControlPointPosition(int(gridVertex-1))
        targetPoints.InsertNextPoint(point)
        sourcePoints.InsertNextPoint(point)
    

    ###################################
    surfacePolydata = meshNode.GetPolyData()

    for gridVertex in gridLandmarks:
      point = LMNode.GetNthControlPointPosition(int(gridVertex-1))
      #targetPoints.InsertNextPoint(point)

    #transform grid to triangle
    transform = vtk.vtkThinPlateSplineTransform()
    transform.SetSourceLandmarks( sourcePoints )
    transform.SetTargetLandmarks( targetPoints )
    transform.SetBasisToR()

    transformNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode","TPS")
    transformNode.SetAndObserveTransformToParent(transform)

    model=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "mesh_resampled")
    model.SetAndObservePolyData(semiPolydata)
    model.SetAndObserveTransformNodeID(transformNode.GetID())
    slicer.vtkSlicerTransformLogic().hardenTransform(model)

    resampledPolydata = model.GetPolyData()

    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(surfacePolydata)
    pointLocator.BuildLocator()

   
    #set up locater for intersection with normal vector rays
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(surfacePolydata)
    obbTree.BuildLocator()

    #define new landmark sets
    semilandmarkNodeName = "semiLM_" + str(gridLandmarks[0]) + "_" + str(gridLandmarks[1]) + "_" + str(gridLandmarks[2])
    semilandmarkPoints=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", semilandmarkNodeName)


    # set initial three grid points
    for index in range(0,3):
      origLMPoint=resampledPolydata.GetPoint(index)
      #landmarkLabel = LMNode.GetNthControlPointLabel(gridLandmarks[index]-1)
      landmarkLabel = str(gridLandmarks[index])
      semilandmarkPoints.AddControlPoint(origLMPoint, landmarkLabel)

    # calculate maximum projection distance
 
    rayLength =  maximumProjectionDistance 
    print("My Raylength is: ", rayLength)
    # get normal projection intersections for remaining semi-landmarks
    my_counter = 0
    
    # Get starting point index (after the 3 grid landmarks)
    current_point_index = 3
    
    for segment_info in num_pointers:
        start_idx = segment_info[0] + current_point_index
        end_idx = segment_info[1] + current_point_index
        
        # Process each point in this segment
        for idx in range(start_idx, end_idx):
            modelPoint = resampledPolydata.GetPoint(idx)

            # Define the vector going fro mmodelpoint -> p0 - which is the corner of the two lines. 
            # Use outline endpoints as reference points for better cross product
            p0 = LMNode.GetNthControlPointPosition(int(gridLandmarks[0]-1))
            
            # For vector 2, going fro modelpoint -> p1, where p1 is the end point in the segmeent, closest too the model point. 
            # Take the stard idx, which are the closests 
            if idx - start_idx < end_idx - 1 - idx:
              # Closer to start
              p1 = resampledPolydata.GetPoint(start_idx)
            else:
              # Closer to end (or equidistant)
              p1 = resampledPolydata.GetPoint(end_idx)
                

            # Method 1: Cross product approach
            rayEndPoint1, rayEndPoint2 = calculate_ray_endpoint(p0, p1, modelPoint, rayLength)
            
            intersectionPoints1 = vtk.vtkPoints()
            intersectionPoints2 = vtk.vtkPoints()
            obbTree.IntersectWithLine(modelPoint, rayEndPoint1, intersectionPoints1, vtk.vtkIdList())
            obbTree.IntersectWithLine(modelPoint, rayEndPoint2, intersectionPoints2, vtk.vtkIdList())
            
            # Find best cross product intersection
            closest_cross_point = None
            min_cross_distance = float('inf')
            
            for points in [intersectionPoints1, intersectionPoints2]:
                len_of_points  = points.GetNumberOfPoints()
                if len_of_points > 0:
                    point = points.GetPoint(len_of_points-1)
                   # point = points.GetPoint(0)
                    dist = vtk.vtkMath().Distance2BetweenPoints(modelPoint, point)
                    if dist < min_cross_distance:
                        min_cross_distance = dist
                        closest_cross_point = point
            
            # Method 2: Direct surface point (fallback)
            closestPointId = pointLocator.FindClosestPoint(modelPoint)
            surface_point = surfacePolydata.GetPoint(closestPointId)
            
            # Decide per point: use cross product if available, otherwise surface point
            if closest_cross_point is not None:
                semilandmarkPoints.AddControlPoint(closest_cross_point)
            else:
                semilandmarkPoints.AddControlPoint(surface_point)
                my_counter += 1 



    print("No intersection at all: ",my_counter)
    # update lock status and color
    semilandmarkPoints.SetLocked(True)
    semilandmarkPoints.GetDisplayNode().SetColor(random.random(), random.random(), random.random())
    semilandmarkPoints.GetDisplayNode().SetSelectedColor(random.random(), random.random(), random.random())
    semilandmarkPoints.GetDisplayNode().PointLabelsVisibilityOff()
    #clean up
    slicer.mrmlScene.RemoveNode(transformNode)
    slicer.mrmlScene.RemoveNode(model)
    print("Total points:", semilandmarkPoints.GetNumberOfControlPoints() )
    return semilandmarkPoints



  def getSmoothNormals(self, surfaceNode,iterations):
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(surfaceNode.GetPolyData())
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.SetNumberOfIterations(iterations)
    smoothFilter.SetRelaxationFactor(1)
    smoothFilter.Update()
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.SetInputData(smoothFilter.GetOutput())
    normalGenerator.Update()
    normalArray = normalGenerator.GetOutput().GetPointData().GetArray("Normals")
    return normalArray

  def getLandmarks(self, landmarkDirectory):
    files_to_open=[]
    for path, dir, files in os.walk(landmarkDirectory):
      for filename in files:
        if fnmatch.fnmatch(filename,"*.fcsv"):
          files_to_open.append(filename)

    tmp = self.readLandmarkFile(os.path.join(landmarkDirectory, files_to_open[0]))    #check size of landmark data array

    [i,j]=tmp.shape
    landmarkArray=np.zeros(shape=(i,j,len(files_to_open)))  # initialize landmark array

    for i in range(len(files_to_open)):
      tmp = self.readLandmarkFile(os.path.join(landmarkDirectory, files_to_open[i]))
      landmarkArray[:,:,i]=tmp
    return landmarkArray

  def readLandmarkFile(self, landmarkFilename):
    datafile=open(landmarkFilename)
    data=[]
    for row in datafile:
      if not fnmatch.fnmatch(row[0],"#*"):  #if not a commented line
        data.append(row.strip().split(','))
    dataArray=np.zeros(shape=(len(data),3))
    j=0
    sorter=[]
    for i in data:
      tmp=np.array(i)[1:4]
      dataArray[j,0:3]=tmp
      x=np.array(i).shape
      j=j+1
    slicer.app.processEvents()
    return dataArray

  def getGridPoints(self, landmarkArray, gridLandmarkNumbers):
    gridArray=[]
    return gridArray

  def setAllLandmarksType(self,landmarkNode, setToSemiType):
    landmarkDescription = "Semi"
    if setToSemiType is False:
      landmarkDescription = "Fixed"
    for controlPointIndex in range(landmarkNode.GetNumberOfControlPoints()):
      landmarkNode.SetNthControlPointDescription(controlPointIndex, landmarkDescription)

  def mergeTree(self, treeWidget, landmarkNode, modelNode,rowColNumber):
    nodeIDs=treeWidget.selectedIndexes()
    nodeList = vtk.vtkCollection()
    for id in nodeIDs:
      if id.column() == 0:
        currentNode = slicer.util.getNode(id.data())
        nodeList.AddItem(currentNode)
    mergedNodeName = landmarkNode.GetName() + "_mergedNode"
    mergedNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', mergedNodeName)
    self.mergeList(nodeList, landmarkNode, modelNode,rowColNumber,mergedNode)
    return True

  def mergeList(self, nodeList, landmarkNode, modelNode,rowColNumber,mergedNode):
    pt=[0,0,0]
    triangleList=[]
    lineSegmentList=[]
    pointList=[]

    # Add semi-landmark points within triangle patches
    for currentNode in nodeList:
      if currentNode != landmarkNode:
        for index in range(3,currentNode.GetNumberOfControlPoints()):
          pt = currentNode.GetNthControlPointPosition(index)
          fiducialLabel = currentNode.GetNthControlPointLabel(index)
          mergedNode.AddControlPoint(pt,fiducialLabel)
        p1=currentNode.GetNthControlPointLabel(0)
        p2=currentNode.GetNthControlPointLabel(1)
        p3=currentNode.GetNthControlPointLabel(2)
        landmarkVector = [p1,p2,p3]
        triangleList.append(landmarkVector)
        lineSegmentList.append(sorted([p1,p2]))
        lineSegmentList.append(sorted([p2,p3]))
        lineSegmentList.append(sorted([p3,p1]))
        pointList.append(p1)
        pointList.append(p2)
        pointList.append(p3)

    # Add semilandmark points on curves between landmark points
    seenVertices=set()
    lineSegmentList_edit=[]
    # Remove duplicate line segments
    for segment in lineSegmentList:
      segmentTuple = tuple(segment)
      if segmentTuple not in seenVertices:
        lineSegmentList_edit.append(segment)
        seenVertices.add(segmentTuple)

    seenPoints=set()
    pointList_edit=[]
    # Remove duplicate points
    for originalLandmarkPoint in pointList:
      lmTuple = tuple(originalLandmarkPoint)
      if lmTuple not in seenPoints:
        pointList_edit.append(originalLandmarkPoint)
        seenPoints.add(lmTuple)

    # add line segments between triangles
    controlPoint=vtk.vtkVector3d()
    edgePoints=vtk.vtkPoints()
    tempCurve = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode', 'temporaryCurve')
    for segment in lineSegmentList_edit:
      landmarkIndex = int(segment[0])
      landmarkNode.GetNthControlPointPosition(int(landmarkIndex-1),controlPoint)
      tempCurve.AddControlPoint(controlPoint)
      landmarkIndex = int(segment[1])
      landmarkNode.GetNthControlPointPosition(int(landmarkIndex-1),controlPoint)
      tempCurve.AddControlPoint(controlPoint)
      sampleDist = tempCurve.GetCurveLengthWorld() / (rowColNumber - 1);
      tempCurve.SetAndObserveSurfaceConstraintNode(modelNode)
      tempCurve.ResampleCurveWorld(sampleDist)
      tempCurve.GetControlPointPositionsWorld(edgePoints)
      for i in range(1,edgePoints.GetNumberOfPoints()-1):
        mergedNode.AddControlPoint(edgePoints.GetPoint(i))
      tempCurve.RemoveAllControlPoints()

    # ------ removing manual points from SL set, leaving this as a placeholder while testing
    # add original landmark points
    #for originalLandmarkPoint in pointList_edit:
    #  landmarkIndex = int(originalLandmarkPoint)
    #  landmarkNode.GetNthControlPointPosition(int(landmarkIndex-1),controlPoint)
    #  mergedNode.AddControlPoint(controlPoint)

    # update lock status and color of merged node
    mergedNode.SetLocked(True)
    mergedNode.GetDisplayNode().SetColor(random.random(), random.random(), random.random())
    mergedNode.GetDisplayNode().SetSelectedColor(random.random(), random.random(), random.random())
    mergedNode.GetDisplayNode().PointLabelsVisibilityOff()
    landmarkTypeSemi=True
    self.setAllLandmarksType(mergedNode, landmarkTypeSemi)

    # clean up
    slicer.mrmlScene.RemoveNode(tempCurve)

    # write selected triangles to table
    tableNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTableNode', 'Semi-landmark Grid')
    col1=tableNode.AddColumn()
    col1.SetName('Vertice 1')
    col2=tableNode.AddColumn()
    col2.SetName('Vertice 2')
    col3=tableNode.AddColumn()
    col3.SetName('Vertice 3')
    tableNode.SetColumnType('Vertice 1',vtk.VTK_STRING)
    tableNode.SetColumnType('Vertice 2',vtk.VTK_STRING)
    tableNode.SetColumnType('Vertice 3',vtk.VTK_STRING)

    for i in range(len(triangleList)):
      tableNode.AddEmptyRow()
      triangle=triangleList[i]
      tableNode.SetCellText(i,0,str(triangle[0]))
      tableNode.SetCellText(i,1,str(triangle[1]))
      tableNode.SetCellText(i,2,str(triangle[2]))

    return True

  def projectPoints(self, sourceMesh, targetMesh, originalPoints, projectedPoints, rayLength):
    sourcePolydata = sourceMesh.GetPolyData()
    targetPolydata = targetMesh.GetPolyData()

    #set up locater for intersection with normal vector rays
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(targetPolydata)
    obbTree.BuildLocator()

    #set up point locator for finding surface normals and closest point
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(sourcePolydata)
    pointLocator.BuildLocator()

    targetPointLocator = vtk.vtkPointLocator()
    targetPointLocator.SetDataSet(targetPolydata)
    targetPointLocator.BuildLocator()

    #get surface normal from each landmark point
    rayDirection=[0,0,0]
    normalArray = sourcePolydata.GetPointData().GetArray("Normals")
    if(not normalArray):
      normalFilter=vtk.vtkPolyDataNormals()
      normalFilter.ComputePointNormalsOn()
      normalFilter.SetInputData(sourcePolydata)
      normalFilter.Update()
      normalArray = normalFilter.GetOutput().GetPointData().GetArray("Normals")
      if(not normalArray):
        print("Error: no normal array")

    for index in range(originalPoints.GetNumberOfControlPoints()):
      originalPoint = originalPoints.GetNthControlPointPosition(index)
      # get ray direction from closest normal
      closestPointId = pointLocator.FindClosestPoint(originalPoint)
      rayDirection = normalArray.GetTuple(closestPointId)
      rayEndPoint=[0,0,0]
      for dim in range(len(rayEndPoint)):
        rayEndPoint[dim] = originalPoint[dim] + rayDirection[dim]* rayLength
      intersectionIds=vtk.vtkIdList()
      intersectionPoints=vtk.vtkPoints()
      obbTree.IntersectWithLine(originalPoint,rayEndPoint,intersectionPoints,intersectionIds)
      #if there are intersections, update the point to most external one.
      if intersectionPoints.GetNumberOfPoints() > 0:
        exteriorPoint = intersectionPoints.GetPoint(intersectionPoints.GetNumberOfPoints()-1)
        projectedPoints.AddControlPoint(exteriorPoint)
      #if there are no intersections, reverse the normal vector
      else:
        for dim in range(len(rayEndPoint)):
          rayEndPoint[dim] = originalPoint[dim] + rayDirection[dim]* -rayLength
        obbTree.IntersectWithLine(originalPoint,rayEndPoint,intersectionPoints,intersectionIds)
        if intersectionPoints.GetNumberOfPoints()>0:
          exteriorPoint = intersectionPoints.GetPoint(0)
          projectedPoints.AddControlPoint(exteriorPoint)
        #if none in reverse direction, use closest mesh point
        else:
          closestPointId = targetPointLocator.FindClosestPoint(originalPoint)
          rayOrigin = targetPolydata.GetPoint(closestPointId)
          projectedPoints.AddControlPoint(rayOrigin)
    return True

  def projectPointsOut(self, sourcePolydata, targetPolydata, originalPoints, projectedPoints, rayLength):
    #sourcePolydata = sourceMesh.GetPolyData()
    #targetPolydata = targetMesh.GetPolyData()

    #set up locater for intersection with normal vector rays
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(targetPolydata)
    obbTree.BuildLocator()

    #set up point locator for finding surface normals and closest point
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(sourcePolydata)
    pointLocator.BuildLocator()

    targetPointLocator = vtk.vtkPointLocator()
    targetPointLocator.SetDataSet(targetPolydata)
    targetPointLocator.BuildLocator()

    #get surface normal from each landmark point
    rayDirection=[0,0,0]
    normalArray = sourcePolydata.GetPointData().GetArray("Normals")
    if(not normalArray):
      normalFilter=vtk.vtkPolyDataNormals()
      normalFilter.ComputePointNormalsOn()
      normalFilter.SetInputData(sourcePolydata)
      normalFilter.Update()
      normalArray = normalFilter.GetOutput().GetPointData().GetArray("Normals")
      if(not normalArray):
        print("Error: no normal array")

    for index in range(originalPoints.GetNumberOfControlPoints()):
      originalPoint = originalPoints.GetNthControlPointPosition(index)
      # get ray direction from closest normal
      closestPointId = pointLocator.FindClosestPoint(originalPoint)
      rayDirection = normalArray.GetTuple(closestPointId)
      rayEndPoint=[0,0,0]
      for dim in range(len(rayEndPoint)):
        rayEndPoint[dim] = originalPoint[dim] + rayDirection[dim]* rayLength
      intersectionIds=vtk.vtkIdList()
      intersectionPoints=vtk.vtkPoints()
      obbTree.IntersectWithLine(originalPoint,rayEndPoint,intersectionPoints,intersectionIds)
      #if there are intersections, update the point to most external one.
      if intersectionPoints.GetNumberOfPoints() > 0:
        exteriorPoint = intersectionPoints.GetPoint(intersectionPoints.GetNumberOfPoints()-1)
        projectedPoints.AddControlPoint(exteriorPoint)
    return True

  def projectPointsOutIn(self, sourcePolydata, targetPolydata, originalPoints, projectedPoints, rayLength):
    #set up locater for intersection with normal vector rays
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(targetPolydata)
    obbTree.BuildLocator()

    #set up point locator for finding surface normals and closest point
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(sourcePolydata)
    pointLocator.BuildLocator()

    targetPointLocator = vtk.vtkPointLocator()
    targetPointLocator.SetDataSet(targetPolydata)
    targetPointLocator.BuildLocator()

    #get surface normal from each landmark point
    rayDirection=[0,0,0]
    normalArray = sourcePolydata.GetPointData().GetArray("Normals")
    if(not normalArray):
      normalFilter=vtk.vtkPolyDataNormals()
      normalFilter.ComputePointNormalsOn()
      normalFilter.SetInputData(sourcePolydata)
      normalFilter.Update()
      normalArray = normalFilter.GetOutput().GetPointData().GetArray("Normals")
      if(not normalArray):
        print("Error: no normal array")

    for index in range(originalPoints.GetNumberOfControlPoints()):
      originalPoint = originalPoints.GetNthControlPointPosition(index)
      # get ray direction from closest normal
      closestPointId = pointLocator.FindClosestPoint(originalPoint)
      rayDirection = normalArray.GetTuple(closestPointId)
      rayEndPoint=[0,0,0]
      for dim in range(len(rayEndPoint)):
        rayEndPoint[dim] = originalPoint[dim] + rayDirection[dim]* rayLength
      intersectionIds=vtk.vtkIdList()
      intersectionPoints=vtk.vtkPoints()
      obbTree.IntersectWithLine(originalPoint,rayEndPoint,intersectionPoints,intersectionIds)
      #if there are intersections, update the point to most external one.
      if intersectionPoints.GetNumberOfPoints() > 0:
        exteriorPoint = intersectionPoints.GetPoint(intersectionPoints.GetNumberOfPoints()-1)
        projectedPoints.AddControlPoint(exteriorPoint)
      #if there are no intersections, reverse the normal vector
      else:
        for dim in range(len(rayEndPoint)):
          rayEndPoint[dim] = originalPoint[dim] + rayDirection[dim]* -rayLength
        obbTree.IntersectWithLine(originalPoint,rayEndPoint,intersectionPoints,intersectionIds)
        if intersectionPoints.GetNumberOfPoints()>0:
          exteriorPoint = intersectionPoints.GetPoint(0)
          projectedPoints.AddControlPoint(exteriorPoint)
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qimage = ctk.ctkWidgetsUtils.grabWidget(widget)
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)

  def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """

    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    cliParams = {
      'InputVolume': inputVolume.GetID(),
      'OutputVolume': outputVolume.GetID(),
      'ThresholdValue' : imageThreshold,
      'ThresholdType' : 'Above' if invert else 'Below'
      }
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    slicer.mrmlScene.RemoveNode(cliNode)

    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


class CreateSemiLMPatchesTest(ScriptedLoadableModuleTest):
  """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
      """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
      """
    self.setUp()
    self.test_CreateSemiLMPatches1()

  def test_CreateSemiLMPatches1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('TemplateKey1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = CreateSemiLMPatchesLogic()

    # Test algorithm with non-inverted threshold
    logic.process(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    # Test algorithm with inverted threshold
    logic.process(inputVolume, outputVolume, threshold, False)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], inputScalarRange[1])

    self.delayDisplay('Test passed')
