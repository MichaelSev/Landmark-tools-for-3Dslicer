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

class LPatch(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "LPatch" # TODO make this more human readable by adding spaces
    self.parent.categories = ["LTools"]
    self.parent.dependencies = []
    self.parent.contributors = ["Michael Lind Severinsen(CCEM)"] # replace with "Firstname Lastname (Organization)"
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

class LPatchWidget(ScriptedLoadableModuleWidget):
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
    gridPointsOutline = qt.QGridLayout()
    
    # Create 3 text input fields for comma-separated numbers
    self.outlinePointsInput1 = qt.QLineEdit()
    self.outlinePointsInput1.setPlaceholderText("1. Enter comma-separated landmark numbers")
    self.outlinePointsInput1.setToolTip("Input comma-separated landmark numbers for first outline (e.g. 1,2,3,4,5)")
    self.outlinePointsInput1.setValidator(qt.QRegExpValidator(qt.QRegExp("[0-9,]*")))
    #self.outlinePointsInput1.setText("1,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,4,2,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,1")  # Set initial value to empty string
    #self.outlinePointsInput1.setText("1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99 ,100 ,101 ,102 ,103 ,104 ,105,106 ,107 ,108 ,109 ,110 ,111 ,112 ,113 ,114 ,115 ,116 ,117 ,118 ,119 ,120 ,121 ,122 ,123 ,124 ,125 ,126 ,127 ,128 ,129 ,130 ,131 ,132 ,133 ,134 ,135 ,136 ,137 ,138 ,139 ,140,141 ,142 ,143 ,144 ,145 ,146 ,147 ,148 ,149 ,150 ,151 ,152,153 ,154 ,155 ,156 ,157 ,158 ,159 ,160 ,161 ,162 ,163 ,164 ,165 ,166 ,167 ,168,169 ,170 ,171 ,172 ,173 ,174 ,175 ,176 ,177 ,178 ,179 ,180 ,181 ,182 ,183 ,184,185 ,186 ,187 ,188")  # Set initial value to empty string
    self.outlinePointsInput1.setText("1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70")  # Set initial value to empty string
  




    # Add the three input fields to the layout
    gridPointsOutline.addWidget(self.outlinePointsInput1, 1, 5)

 
    
    parametersFormLayout.addRow("Outline:", gridPointsOutline)

    # Use entire landmark line checkbox
    self.useEntireLandmarkLineCheckBox = qt.QCheckBox()
    self.useEntireLandmarkLineCheckBox.checked = 0
    self.useEntireLandmarkLineCheckBox.setToolTip("If checked, use all landmarks in the landmark line instead of specific indices")
    parametersFormLayout.addRow("Use entire landmark line:", self.useEntireLandmarkLineCheckBox)

    #
    # Point placement method selection
    #
    self.placementMethodGroup = qt.QButtonGroup()
    placementMethodLayout = qt.QHBoxLayout()
    
    self.pointCountRadio = qt.QRadioButton("By point count")
    self.pointCountRadio.setChecked(True)
    self.pointCountRadio.setToolTip("Place a specific number of points evenly along the outline")
    self.placementMethodGroup.addButton(self.pointCountRadio, 0)
    placementMethodLayout.addWidget(self.pointCountRadio)
    
    self.distanceRadio = qt.QRadioButton("By distance")
    self.distanceRadio.setToolTip("Place points at a specific distance interval along the outline")
    self.placementMethodGroup.addButton(self.distanceRadio, 1)
    placementMethodLayout.addWidget(self.distanceRadio)
    
    parametersFormLayout.addRow("Placement method:", placementMethodLayout)
    
    #
    # Point count input (original)
    #
    self.gridSamplingRate = ctk.ctkDoubleSpinBox()
    self.gridSamplingRate.minimum = 3
    self.gridSamplingRate.maximum = 500
    self.gridSamplingRate.singleStep = 1
    self.gridSamplingRate.setDecimals(0)
    self.gridSamplingRate.value = 10
    self.gridSamplingRate.setToolTip("Number of points to place evenly along the outline")
    parametersFormLayout.addRow("Number of points:", self.gridSamplingRate)
    
    #
    # Distance input (new)
    #
    self.distanceSpacing = ctk.ctkDoubleSpinBox()
    self.distanceSpacing.minimum = 0.1
    self.distanceSpacing.maximum = 100.0
    self.distanceSpacing.singleStep = 0.1
    self.distanceSpacing.setDecimals(1)
    self.distanceSpacing.value = 2.0
    self.distanceSpacing.enabled = False  # Initially disabled
    self.distanceSpacing.setToolTip("Distance between consecutive points along the outline")
    parametersFormLayout.addRow("Point spacing distance:", self.distanceSpacing)

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
    # Optimization penalty weights
    #
    penaltyGroupBox = ctk.ctkCollapsibleGroupBox()
    penaltyGroupBox.setTitle("Optimization Penalty Weights")
    penaltyGroupBox.collapsed = True
    advancedFormLayout.addRow(penaltyGroupBox)
    penaltyLayout = qt.QFormLayout(penaltyGroupBox)

    # Original distance penalty weight
    self.originalDistancePenalty = ctk.ctkDoubleSpinBox()
    self.originalDistancePenalty.minimum = 0.0
    self.originalDistancePenalty.maximum = 100.0
    self.originalDistancePenalty.singleStep = 0.001
    self.originalDistancePenalty.setDecimals(2)
    self.originalDistancePenalty.value = 1.0
    self.originalDistancePenalty.setToolTip("Weight for keeping points close to original interpolated positions")
    penaltyLayout.addRow("Original distance penalty:", self.originalDistancePenalty)

    # Surface distance penalty weight
    self.surfaceDistancePenalty = ctk.ctkDoubleSpinBox()
    self.surfaceDistancePenalty.minimum = 0.0
    self.surfaceDistancePenalty.maximum = 100.0
    self.surfaceDistancePenalty.singleStep = 0.1
    self.surfaceDistancePenalty.setDecimals(2)
    self.surfaceDistancePenalty.value = 10.0
    self.surfaceDistancePenalty.setToolTip("Weight for encouraging proximity to surface")
    penaltyLayout.addRow("Surface distance penalty:", self.surfaceDistancePenalty)

    # Spacing penalty weight
    self.spacingPenalty = ctk.ctkDoubleSpinBox()
    self.spacingPenalty.minimum = 0.0
    self.spacingPenalty.maximum = 100.0
    self.spacingPenalty.singleStep = 0.1
    self.spacingPenalty.setDecimals(2)
    self.spacingPenalty.value = 1.0
    self.spacingPenalty.setToolTip("Weight for maintaining even spacing between points")
    penaltyLayout.addRow("Spacing penalty:", self.spacingPenalty)

    # Smoothness penalty weight
    self.smoothnessPenalty = ctk.ctkDoubleSpinBox()
    self.smoothnessPenalty.minimum = 0.0
    self.smoothnessPenalty.maximum = 100.0
    self.smoothnessPenalty.singleStep = 0.01
    self.smoothnessPenalty.setDecimals(2)
    self.smoothnessPenalty.value = 1.0
    self.smoothnessPenalty.setToolTip("Weight for maintaining smooth transitions between points")
    penaltyLayout.addRow("Smoothness penalty:", self.smoothnessPenalty)

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
    self.pointCountRadio.connect('toggled(bool)', self.onPlacementMethodChanged)
    self.distanceRadio.connect('toggled(bool)', self.onPlacementMethodChanged)
    self.useEntireLandmarkLineCheckBox.connect('toggled(bool)', self.onUseEntireLandmarkLineChanged)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass

  def onPlacementMethodChanged(self):
    """Enable/disable input fields based on placement method selection."""
    if self.pointCountRadio.isChecked():
      self.gridSamplingRate.enabled = True
      self.distanceSpacing.enabled = False
    else:  # distance radio is checked
      self.gridSamplingRate.enabled = False
      self.distanceSpacing.enabled = True

  def onUseEntireLandmarkLineChanged(self):
    """Enable/disable outline input field based on checkbox state."""
    if self.useEntireLandmarkLineCheckBox.checked:
      self.outlinePointsInput1.enabled = False
      self.outlinePointsInput1.setPlaceholderText("Using entire landmark line - input disabled")
    else:
      self.outlinePointsInput1.enabled = True
      self.outlinePointsInput1.setPlaceholderText("1. Enter comma-separated landmark numbers")

  def onApplyButton(self):
    logic = CreateSemiLMPatchesLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    
    # Get outline landmark indices with error handling
    if self.useEntireLandmarkLineCheckBox.checked:
        # Use entire landmark line
        if not self.LMSelect.currentNode():
            slicer.util.errorDisplay("Please select a landmark node")
            return False
        
        total_landmarks = self.LMSelect.currentNode().GetNumberOfControlPoints()
        if total_landmarks < 2:
            slicer.util.errorDisplay("Landmark node must have at least 2 landmarks")
            return False
        
        outline = list(range(1, total_landmarks + 1))  # 1-based indexing
        print(f"Using entire landmark line: {len(outline)} landmarks")
    else:
        # Use specific indices from input field
        outline_text = self.outlinePointsInput1.text.strip()
        
        if not outline_text:
            slicer.util.errorDisplay("Please enter outline landmark indices (comma-separated numbers)")
            return False
        
        try:
            outline = [int(x.strip()) for x in outline_text.split(",") if x.strip()]
        except ValueError as e:
            slicer.util.errorDisplay(f"Invalid outline format. Please enter comma-separated numbers. Error: {str(e)}")
            return False
        
        if len(outline) < 2:
            slicer.util.errorDisplay("Please enter at least 2 outline landmark indices")
            return False

    # Determine placement method and value
    use_point_count = self.pointCountRadio.isChecked()
    if use_point_count:
      placement_value = int(self.gridSamplingRate.value) + 1
    else:
      placement_value = self.distanceSpacing.value

    # Get penalty weights
    penalty_weights = {
      'original_distance': self.originalDistancePenalty.value,
      'surface_distance': self.surfaceDistancePenalty.value,
      'spacing': self.spacingPenalty.value,
      'smoothness': self.smoothnessPenalty.value
    }

    logic.run(self.meshSelect.currentNode(), self.LMSelect.currentNode(), outline, placement_value, use_point_count, penalty_weights)

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
  def run(self, meshNode, LMNode, outline, placementValue, usePointCount=True, penaltyWeights=None):
   
    semiLandmarks = self.applyPatch(meshNode, LMNode, outline, placementValue, usePointCount, penaltyWeights)

    return True


  def applyPatch(self, meshNode, LMNode, outline, placementValue, usePointCount=True, penaltyWeights=None):
    
    if usePointCount:
        print(f"Outline: {outline}, Point count: {int(placementValue)}")
    else:
        print(f"Outline: {outline}, Distance spacing: {placementValue}")
    
    import numpy as np
    import vtk
    import slicer
    surface_polydata = meshNode.GetPolyData()
     
    # ==================== MAIN PROCESSING ====================
    
    # Extract existing landmark points directly (no interpolation)
    outline_positions = []
    max_landmarks = LMNode.GetNumberOfControlPoints()
    
    for lm_idx in outline:
        idx = int(lm_idx - 1)  # Convert to 0-based index
        if idx < 0 or idx >= max_landmarks:
            print(f"Warning: Landmark index {lm_idx} is out of range (1-{max_landmarks}). Skipping.")
            continue
        
        point = LMNode.GetNthControlPointPosition(idx)
        if point is None:
            print(f"Warning: Could not get position for landmark {lm_idx}. Skipping.")
            continue
        
        outline_positions.append(list(point))
    
    if len(outline_positions) < 2:
        print(f"Error: Need at least 2 valid outline landmark points, got {len(outline_positions)}")
        return None
    
    outline_positions = np.array(outline_positions)
    print(f"Using {len(outline_positions)} existing outline landmark points as boundary")
    
    # ==================== LAPLACE SOLVE FOR SURFACE DEFINITION ====================
    

   

    def simple_sampling(sample_points, outline_points, target_count):
        """
        Simple and clean point sampling using farthest point sampling.
        Prioritizes points that are far from already sampled points while respecting boundary constraints.
        
        Args:
            sample_points: List or array of 3D points to sample from
            outline_points: Outline/boundary points (for distance constraints)
            target_count: Number of points to keep
        
        Returns:
            List of sampled points
        """
        import numpy as np
        
        sample_points = np.array(sample_points)
        outline_points = np.array(outline_points)
        n_points = len(sample_points)
        
        print(f"Sampling from {n_points} points, target: {target_count}")
        
        if target_count >= n_points:
            return sample_points.tolist()
        
        sampled_indices = []
        remaining_indices = list(range(n_points))
        
        # Choose starting point - pick one closest to center of outline
        outline_center = np.mean(outline_points, axis=0)
        distances_to_outline_center = [
            np.linalg.norm(sample_points[idx] - outline_center) 
            for idx in remaining_indices
        ]
        start_idx = remaining_indices[np.argmin(distances_to_outline_center)]
        
        sampled_indices.append(start_idx)
        remaining_indices.remove(start_idx)
        
        # Iteratively add the farthest point from all sampled points
        while len(sampled_indices) < target_count and remaining_indices:
            max_min_distance = -1
            best_idx = None
            
            for idx in remaining_indices:
                candidate = sample_points[idx]
                
                # Find minimum distance to any already sampled point
                min_dist_to_sampled = min([
                    np.linalg.norm(candidate - sample_points[sampled_idx])
                    for sampled_idx in sampled_indices
                ])
                
                effective_distance = min_dist_to_sampled
                
                if effective_distance > max_min_distance:
                    max_min_distance = effective_distance
                    best_idx = idx
            
            if best_idx is not None:
               # print(f"Selected point with distance: {max_min_distance:.3f}")
                sampled_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        # Return the actual sampled points
        sampled_points = sample_points[sampled_indices]
        return sampled_points.tolist()

    def distance_based_sampling(sample_points, outline_points, target_distance):
        """
        Distance-based point sampling that keeps adding points until no more points 
        can be added with the minimum target distance between them.
        
        Args:
            sample_points: List or array of 3D points to sample from
            outline_points: Outline/boundary points (for reference)
            target_distance: Minimum distance required between selected points
        
        Returns:
            List of sampled points
        """
        import numpy as np
        
        sample_points = np.array(sample_points)
        outline_points = np.array(outline_points)
        n_points = len(sample_points)
        
        print(f"Distance-based sampling from {n_points} points, target distance: {target_distance}")
        
        sampled_indices = []
        remaining_indices = list(range(n_points))
        
        # Choose starting point - pick one closest to center of outline
        outline_center = np.mean(outline_points, axis=0)
        distances_to_outline_center = [
            np.linalg.norm(sample_points[idx] - outline_center) 
            for idx in remaining_indices
        ]
        start_idx = remaining_indices[np.argmin(distances_to_outline_center)]
        
        sampled_indices.append(start_idx)
        remaining_indices.remove(start_idx)
        
        # Keep adding points until no more points can satisfy the distance constraint
        points_added = True
        while points_added and remaining_indices:
            points_added = False
            best_idx = None
            max_min_distance = -1
            
            # Find the point that has the largest minimum distance to all sampled points
            # and satisfies the target distance constraint
            for idx in remaining_indices:
                candidate = sample_points[idx]
                
                # Find minimum distance to any already sampled point
                min_dist_to_sampled = min([
                    np.linalg.norm(candidate - sample_points[sampled_idx])
                    for sampled_idx in sampled_indices
                ])
                
                # Only consider points that satisfy the minimum distance constraint
                if min_dist_to_sampled >= target_distance:
                    if min_dist_to_sampled > max_min_distance:
                        max_min_distance = min_dist_to_sampled
                        best_idx = idx
            
            # If we found a valid point, add it
            if best_idx is not None:
                #print(f"Added point with minimum distance: {max_min_distance:.3f}")
                sampled_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                points_added = True
        
        print(f"Distance-based sampling completed with {len(sampled_indices)} points")
        
        # Return the actual sampled points
        sampled_points = sample_points[sampled_indices]
        return sampled_points.tolist()

    def generate_surface_landmarks_from_outline(outline_points, num_output, iterations=15, use_point_count=True):
        """
        Generate surface landmarks using outline interpolation method:
        1. Estimate center from outline points
        2. For each outline point, interpolate line to center
        3. Place point at progressive distance (1/(iter/total_iter)) along each line
        4. Project points to surface
        5. Re-estimate center and repeat
        
        Args:
            outline_points: Array of outline landmark positions
            num_output: Number of surface points to generate (if use_point_count=True) or minimum distance (if use_point_count=False)
            iterations: Number of center re-estimation iterations
            use_point_count: If True, num_output is target point count; if False, num_output is minimum distance
        
        Returns:
            List of surface landmark positions
        """
        import numpy as np
        
        outline_array = np.array(outline_points, dtype=float)
        current_points = outline_array.copy()
        all_points = []

       # indices = np.linspace(0, len(outline_array) - 1, 6, dtype=int)
       # current_points = outline_array[indices]
   
        point_locator = vtk.vtkPointLocator()
        point_locator.SetDataSet(surface_polydata)
        point_locator.BuildLocator()

        for iteration in range(iterations-1):
            print(f"Iteration {iteration + 1}/{iterations}")

            # Move 1/total_iterations of the total distance toward center
            step_fraction = 1.0 / (iterations-iteration)
            center = np.mean(current_points, axis=0)
            directions = center - current_points
            intermediate_points = current_points + step_fraction * directions
            projected_points = intermediate_points     

            
            # Project intermediate points to surface
            projected_points = []         
            for point in intermediate_points:
                closest_id = point_locator.FindClosestPoint(point)
                surface_point = surface_polydata.GetPoint(closest_id)
                projected_points.append(list(surface_point))
            
        
            # add the surface landmarks to the output
            if iteration > 1:
                all_points.extend(projected_points)
            current_points = np.array(projected_points)
            
            # Re-estimate center from projected points for next iteration
            # Update center for next iteration

        # Apply appropriate sampling method based on user selection
        if use_point_count:
            sampled_points = simple_sampling(all_points, outline_array, num_output)
        else:
            # Use distance-based sampling
            sampled_points = distance_based_sampling(all_points, outline_array, num_output)

        
        return np.array(sampled_points)
    

    surface_points = generate_surface_landmarks_from_outline(outline_positions, placementValue, 30, usePointCount)

    # fix surfaace_points shape 
    if surface_points.ndim == 1:
        surface_points = surface_points.reshape(-1, 3)

    # ==================== CREATE FINAL LANDMARKS ====================
    
    # ==================== OPTIMIZE SURFACE POINTS TO SURFACE ====================
    
    def optimize_points_to_surface(inputpoints, surface_polydata, penalty_weights):
        """Optimize surface points to lie exactly on the mesh surface using scipy optimization."""
        from scipy.optimize import minimize
        import numpy as np
        
        # Create VTK point locator for surface projection
        point_locator = vtk.vtkPointLocator()
        point_locator.SetDataSet(surface_polydata)
        point_locator.BuildLocator()
        
       # Final projection to ensure points are exactly on surface
        points = []
        for pos in inputpoints:
            closest_id = point_locator.FindClosestPoint(pos)
            surface_point = surface_polydata.GetPoint(closest_id)
            points.append(list(surface_point))
        


        def get_surface_distances(positions):
            """Get distances from points to surface and their closest surface points."""
            distances = []
            closest_points = []
            for pos in positions:
                closest_id = point_locator.FindClosestPoint(pos)
                surface_point = np.array(surface_polydata.GetPoint(closest_id))
                distance = np.linalg.norm(pos - surface_point)
                distances.append(distance)
                closest_points.append(surface_point)
            return np.array(distances), np.array(closest_points)
        
        def optimization_objective(positions_flat):
            """Objective function for surface optimization."""
            positions = positions_flat.reshape(-1, 3)
            
            # Term 1: Stay reasonably close to original positions
            original_positions = np.array(points)
            original_distance_penalty = np.sum((positions - original_positions)**2)
            
            # Term 2: Distance to surface penalty (primary goal)
            surface_distances, closest_surface_points = get_surface_distances(positions)
            surface_penalty = np.sum(surface_distances**2)
            
            # Term 3: Maintain spacing between points
            spacing_penalty = 0.0
            if len(positions) >= 2:
                # For each point, find the distances to its 3 nearest neighbors
                nearest3_distances = []
                for i, pos in enumerate(positions):
                    other_positions = np.delete(positions, i, axis=0)
                    dists = np.linalg.norm(other_positions - pos, axis=1)
                    if len(dists) >= 3:
                        nearest_dists = np.partition(dists, 2)[:3]
                    elif len(dists) > 0:
                        nearest_dists = dists
                    else:
                        nearest_dists = []
                    if len(nearest_dists) > 0:
                        nearest3_distances.extend(nearest_dists)
                nearest3_distances = np.array(nearest3_distances)
                if len(nearest3_distances) > 0:
                    # Penalize points whose 3-nearest neighbor distances deviate from the mean
                    spacing_penalty = np.sum((nearest3_distances - 2) ** 2)
            
            # Term 4: Distance to boundary points (maintain boundary awareness)
            boundary_penalty = 0.0
            if outline_positions is not None and len(outline_positions) > 0:
                for pos in positions:
                    boundary_distances = np.linalg.norm(outline_positions - pos, axis=1)
                    min_boundary_dist = np.min(boundary_distances)
                    # Encourage some minimum distance from boundary
                    if min_boundary_dist < 1.0:  # Adjust threshold as needed
                        boundary_penalty += (1.0 - min_boundary_dist)**2
            
            # Combine penalties
            total_loss = (
                penalty_weights.get('original_distance', 1.0) * original_distance_penalty +  # Stay near original
                penalty_weights.get('surface_distance', 1.0) * surface_penalty +              # Encourage proximity to surface
                penalty_weights.get('spacing', 1.0) * spacing_penalty +                       # Maintain even spacing
                penalty_weights.get('boundary_awareness', 1) * boundary_penalty             # Maintain boundary awareness
            )
            
            return total_loss
        
        # Convert points to numpy array and flatten for optimization
        initial_positions = np.array(points)
        positions_flat = initial_positions.flatten()
        
        # Run optimization
        result = minimize(
            optimization_objective,
            positions_flat,
            method='L-BFGS-B',
            options={
                'maxiter': 1,
                'disp': False,
                'ftol': 1e-9,
                'gtol': 1e-6
            }
        )
        
        # Reshape optimized positions back to 3D points
        optimized_positions = result.x.reshape(-1, 3)
        
 
        return optimized_positions

    # Create semilandmark node for 3D surface sampled points
    #if surface_points:
    # Optimize surface points to lie exactly on the mesh surface
    optimized_surface_points = optimize_points_to_surface(
         surface_points, 
         surface_polydata, 
         penaltyWeights
     )
    
    surface_points = optimized_surface_points
    surface_node_name = f"semiLM_Outline_Interpolation_{len(surface_points)}pts"
    surface_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", surface_node_name)
    
    # Add surface landmark points
    for i, surface_point in enumerate(surface_points):
        label = f"outline_interp_{i+1}"
        surface_node.AddControlPoint(surface_point, label)
    
    # Set display properties
    surface_node.SetLocked(True)
    surface_display_node = surface_node.GetDisplayNode()
    random_color = [random.random(), random.random(), random.random()]
    surface_display_node.SetColor(*random_color)
    # Set a slightly darker random color for selected color
    selected_color = [max(0, c - 0.2) for c in random_color]
    surface_display_node.SetSelectedColor(*selected_color)
    surface_display_node.PointLabelsVisibilityOff()
    
    print(f"Created {len(surface_points)} surface landmarks using outline interpolation method")
    

    return surface_node


