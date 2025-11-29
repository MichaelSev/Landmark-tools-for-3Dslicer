import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import fnmatch
import  numpy as np
import random
import math

# Jaxgeometry and landmarks imports for diffeomorphism estimation
import jax
import jax.numpy as jnp
import jaxgeometry

from jaxgeometry.manifolds.landmarks import *   


#
# CreateSemiLMPatches
#

class LTransfer(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "LTransfer" # TODO make this more human readable by adding spaces
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

class LTransferWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  def onMeshSelect(self):
    self.applyButton.enabled = bool (self.meshSourceSelect.currentNode() and self.meshTargetSelect.currentNode() and self.LMSourceSelect.currentNode() and self.LMTargetSelect.currentNode())
    nodes=self.fiducialView.selectedIndexes()
    self.mergeButton.enabled = bool (nodes and self.LMSourceSelect.currentNode() and self.meshSourceSelect.currentNode() and self.LMTargetSelect.currentNode() and self.meshTargetSelect.currentNode())

  def onLMSelect(self):
    self.applyButton.enabled = bool (self.meshSourceSelect.currentNode() and self.meshTargetSelect.currentNode() and self.LMSourceSelect.currentNode() and self.LMTargetSelect.currentNode())
    nodes=self.fiducialView.selectedIndexes()
    self.mergeButton.enabled = bool (nodes and self.LMSourceSelect.currentNode() and self.meshSourceSelect.currentNode() and self.LMTargetSelect.currentNode() and self.meshTargetSelect.currentNode())

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

    # 3D view set up tab - Source model
    self.meshSourceSelect = slicer.qMRMLNodeComboBox()
    self.meshSourceSelect.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.meshSourceSelect.selectNodeUponCreation = False
    self.meshSourceSelect.addEnabled = False
    self.meshSourceSelect.removeEnabled = False
    self.meshSourceSelect.noneEnabled = True
    self.meshSourceSelect.showHidden = False
    self.meshSourceSelect.setMRMLScene( slicer.mrmlScene )
    self.meshSourceSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onMeshSelect)
    parametersFormLayout.addRow("Source Model: ", self.meshSourceSelect)
    self.meshSourceSelect.setToolTip( "Select source model node for semilandmarking" )

    # Target model
    self.meshTargetSelect = slicer.qMRMLNodeComboBox()
    self.meshTargetSelect.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.meshTargetSelect.selectNodeUponCreation = False
    self.meshTargetSelect.addEnabled = False
    self.meshTargetSelect.removeEnabled = False
    self.meshTargetSelect.noneEnabled = True
    self.meshTargetSelect.showHidden = False
    self.meshTargetSelect.setMRMLScene( slicer.mrmlScene )
    self.meshTargetSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onMeshSelect)
    parametersFormLayout.addRow("Target Model: ", self.meshTargetSelect)
    self.meshTargetSelect.setToolTip( "Select target model node for semilandmarking" )

    # Source landmarks
    self.LMSourceSelect = slicer.qMRMLNodeComboBox()
    self.LMSourceSelect.nodeTypes = ( ('vtkMRMLMarkupsFiducialNode'), "" )
    self.LMSourceSelect.selectNodeUponCreation = False
    self.LMSourceSelect.addEnabled = False
    self.LMSourceSelect.removeEnabled = False
    self.LMSourceSelect.noneEnabled = True
    self.LMSourceSelect.showHidden = False
    self.LMSourceSelect.showChildNodeTypes = False
    self.LMSourceSelect.setMRMLScene( slicer.mrmlScene )
    self.LMSourceSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onLMSelect)
    parametersFormLayout.addRow("Source Landmark set: ", self.LMSourceSelect)
    self.LMSourceSelect.setToolTip( "Select the source landmark set that corresponds to the source model" )

    # Target landmarks
    self.LMTargetSelect = slicer.qMRMLNodeComboBox()
    self.LMTargetSelect.nodeTypes = ( ('vtkMRMLMarkupsFiducialNode'), "" )
    self.LMTargetSelect.selectNodeUponCreation = False
    self.LMTargetSelect.addEnabled = False
    self.LMTargetSelect.removeEnabled = False
    self.LMTargetSelect.noneEnabled = True
    self.LMTargetSelect.showHidden = False
    self.LMTargetSelect.showChildNodeTypes = False
    self.LMTargetSelect.setMRMLScene( slicer.mrmlScene )
    self.LMTargetSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onLMSelect)
    parametersFormLayout.addRow("Target Landmark set: ", self.LMTargetSelect)
    self.LMTargetSelect.setToolTip( "Select the target landmark set that corresponds to the target model" )


    #
    # input landmark numbers for grid
    #
    gridPointsOutline = qt.QGridLayout()
    
    # Create text input fields for source and target comma-separated numbers
    self.outlinePointsSourceInput = qt.QLineEdit()
    self.outlinePointsSourceInput.setPlaceholderText("Enter comma-separated source landmark numbers")
    self.outlinePointsSourceInput.setToolTip("Input comma-separated landmark numbers for source outline (e.g. 1,2,3,4,5)")
    self.outlinePointsSourceInput.setValidator(qt.QRegExpValidator(qt.QRegExp("[0-9,]*")))
    #self.outlinePointsSourceInput.setText("1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99 ,100 ,101 ,102 ,103 ,104 ,105,106 ,107 ,108 ,109 ,110 ,111 ,112 ,113 ,114 ,115 ,116 ,117 ,118 ,119 ,120 ,121 ,122 ,123 ,124 ,125 ,126 ,127 ,128 ,129 ,130 ,131 ,132 ,133 ,134 ,135 ,136 ,137 ,138 ,139 ,140,141 ,142 ,143 ,144 ,145 ,146 ,147 ,148 ,149 ,150 ,151 ,152,153 ,154 ,155 ,156 ,157 ,158 ,159 ,160 ,161 ,162 ,163 ,164 ,165 ,166 ,167 ,168,169 ,170 ,171 ,172 ,173 ,174 ,175 ,176 ,177 ,178 ,179 ,180 ,181 ,182 ,183 ,184,185 ,186 ,187 ,188")
    self.outlinePointsSourceInput.setText("1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50")
    self.outlinePointsTargetInput = qt.QLineEdit()
    self.outlinePointsTargetInput.setPlaceholderText("Enter comma-separated target landmark numbers")
    self.outlinePointsTargetInput.setToolTip("Input comma-separated landmark numbers for target outline (e.g. 1,2,3,4,5)")
    self.outlinePointsTargetInput.setValidator(qt.QRegExpValidator(qt.QRegExp("[0-9,]*")))
    #self.outlinePointsTargetInput.setText("1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99 ,100 ,101 ,102 ,103 ,104 ,105,106 ,107 ,108 ,109 ,110 ,111 ,112 ,113 ,114 ,115 ,116 ,117 ,118 ,119 ,120 ,121 ,122 ,123 ,124 ,125 ,126 ,127 ,128 ,129 ,130 ,131 ,132 ,133 ,134 ,135 ,136 ,137 ,138 ,139 ,140,141 ,142 ,143 ,144 ,145 ,146 ,147 ,148 ,149 ,150 ,151 ,152,153 ,154 ,155 ,156 ,157 ,158 ,159 ,160 ,161 ,162 ,163 ,164 ,165 ,166 ,167 ,168,169 ,170 ,171 ,172 ,173 ,174 ,175 ,176 ,177 ,178 ,179 ,180 ,181 ,182 ,183 ,184,185 ,186 ,187 ,188")
    self.outlinePointsTargetInput.setText("1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50")
    # Add the input fields to the layout
    gridPointsOutline.addWidget(self.outlinePointsSourceInput, 0, 0)
    gridPointsOutline.addWidget(self.outlinePointsTargetInput, 1, 0)

 
    
    parametersFormLayout.addRow("Source Outline:", self.outlinePointsSourceInput)
    parametersFormLayout.addRow("Target Outline:", self.outlinePointsTargetInput)

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
      self.outlinePointsSourceInput.enabled = False
      self.outlinePointsTargetInput.enabled = False
      self.outlinePointsSourceInput.setPlaceholderText("Using entire landmark line - input disabled")
      self.outlinePointsTargetInput.setPlaceholderText("Using entire landmark line - input disabled")
    else:
      self.outlinePointsSourceInput.enabled = True
      self.outlinePointsTargetInput.enabled = True
      self.outlinePointsSourceInput.setPlaceholderText("Enter comma-separated source landmark numbers")
      self.outlinePointsTargetInput.setPlaceholderText("Enter comma-separated target landmark numbers")

  def onApplyButton(self):
    logic = CreateSemiLMPatchesLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    
    # Get outline landmark indices with error handling for both source and target
    if self.useEntireLandmarkLineCheckBox.checked:
        # Use entire landmark line for source
        if not self.LMSourceSelect.currentNode():
            slicer.util.errorDisplay("Please select a source landmark node")
            return False
        
        total_source_landmarks = self.LMSourceSelect.currentNode().GetNumberOfControlPoints()
        if total_source_landmarks < 2:
            slicer.util.errorDisplay("Source landmark node must have at least 2 landmarks")
            return False
        
        source_outline = list(range(1, total_source_landmarks + 1))  # 1-based indexing
        print(f"Using entire source landmark line: {len(source_outline)} landmarks")
        
        # Use entire landmark line for target
        if not self.LMTargetSelect.currentNode():
            slicer.util.errorDisplay("Please select a target landmark node")
            return False
        
        total_target_landmarks = self.LMTargetSelect.currentNode().GetNumberOfControlPoints()
        if total_target_landmarks < 2:
            slicer.util.errorDisplay("Target landmark node must have at least 2 landmarks")
            return False
        
        target_outline = list(range(1, total_target_landmarks + 1))  # 1-based indexing
        print(f"Using entire target landmark line: {len(target_outline)} landmarks")
    else:
        # Use specific indices from input fields for source
        source_outline_text = self.outlinePointsSourceInput.text.strip()
        
        if not source_outline_text:
            slicer.util.errorDisplay("Please enter source outline landmark indices (comma-separated numbers)")
            return False
        
        try:
            source_outline = [int(x.strip()) for x in source_outline_text.split(",") if x.strip()]
        except ValueError as e:
            slicer.util.errorDisplay(f"Invalid source outline format. Please enter comma-separated numbers. Error: {str(e)}")
            return False
        
        if len(source_outline) < 2:
            slicer.util.errorDisplay("Please enter at least 2 source outline landmark indices")
            return False
        
        # Use specific indices from input fields for target
        target_outline_text = self.outlinePointsTargetInput.text.strip()
        
        if not target_outline_text:
            slicer.util.errorDisplay("Please enter target outline landmark indices (comma-separated numbers)")
            return False
        
        try:
            target_outline = [int(x.strip()) for x in target_outline_text.split(",") if x.strip()]
        except ValueError as e:
            slicer.util.errorDisplay(f"Invalid target outline format. Please enter comma-separated numbers. Error: {str(e)}")
            return False
        
        if len(target_outline) < 2:
            slicer.util.errorDisplay("Please enter at least 2 target outline landmark indices")
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

    logic.run(self.meshSourceSelect.currentNode(), self.meshTargetSelect.currentNode(), self.LMSourceSelect.currentNode(), self.LMTargetSelect.currentNode(), source_outline, target_outline, placement_value, use_point_count, penalty_weights)

  def onMergeButton(self):
    logic = CreateSemiLMPatchesLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    logic.mergeTree(self.fiducialView, self.LMSourceSelect.currentNode(), self.meshSourceSelect.currentNode(), int(self.gridSamplingRate.value))

  def updateMergeButton(self):
    nodes=self.fiducialView.selectedIndexes()
    self.mergeButton.enabled = bool (nodes and self.LMSourceSelect.currentNode() and self.meshSourceSelect.currentNode() and self.LMTargetSelect.currentNode() and self.meshTargetSelect.currentNode())

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

  def estimate_and_apply(self, source_outline_points, target_outline_points, surface_points):
    import numpy as np
    from scipy.optimize import minimize

    def align_shapes_3d(target_points, source_points, allow_reflection=False):
        """
        Align target_points -> source_points with a similarity transform (scale, rotation, translation).

        Parameters
        ----------
        target_points : (N,3) or flat array
            The points you want to move.
        source_points : (N,3) or flat array
            The reference points you want to align to.
        allow_reflection : bool
            If False (default), reflections are prevented. Set True if you want to allow mirrored solutions.

        Returns
        -------
        aligned_target : (N,3)
            The target_points after alignment into source space.
        transform_params : dict
            Contains rotation matrix, scale, translation, centroids, and RMS error.
        """
        def to_nx3(vec):
            a = np.asarray(vec, dtype=float)
            if a.ndim == 1:
                if a.size % 3 != 0:
                    raise ValueError("Point arrays must be length multiple of 3.")
                return np.column_stack([a[0::3], a[1::3], a[2::3]])
            elif a.ndim == 2 and a.shape[1] == 3:
                return a
            else:
                raise ValueError("Points must be (N,3) or 1D flattened with length multiple of 3.")

        X = to_nx3(target_points)
        Y = to_nx3(source_points)

        n = X.shape[0]
        centroid_X = X.mean(axis=0)
        centroid_Y = Y.mean(axis=0)

        Xc = X - centroid_X
        Yc = Y - centroid_Y

        # covariance
        cov = (Yc.T @ Xc) / n

        # SVD
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            if not allow_reflection:
                S[2, 2] = -1.0

        R = U @ S @ Vt
        scale = np.sum(D * np.diag(S)) / (np.sum(Xc ** 2) / n)
        t = centroid_Y - scale * (centroid_X @ R.T)

        aligned_target = scale * (X @ R.T) + t

        rms = np.sqrt(np.mean(np.sum((aligned_target - Y) ** 2, axis=1)))

        transform_params = {
            "rotation_matrix": R,
            "scale": float(scale),
            "translation": tuple(t.tolist()),
            "centroid_target": centroid_X,
            "centroid_source": centroid_Y,
            "rms_error": float(rms),
        }

        return aligned_target, transform_params


    def align_shapes_3d_with_followers(target_points, source_points, extra_points=None, allow_reflection=False):
        """
        Align target_points -> source_points with a similarity transform (scale, rotation, translation).
        Optionally also transform extra_points along with it, both forward and reverse.

        Parameters
        ----------
        target_points : (N,3) or flat array
            The points you want to move.
        source_points : (N,3) or flat array
            The reference points you want to align to.
        extra_points : (M,3) or flat array, optional
            Extra points you want to transform along with the target.
        allow_reflection : bool
            If False (default), reflections are prevented.

        Returns
        -------
        aligned_target : (N,3)
            The target_points after alignment into source space.
        aligned_extra : (M,3) or None
            The extra_points transformed forward into source space.
        restored_target : (N,3)
            The target_points transformed back from source space.
        restored_extra : (M,3) or None
            The extra_points transformed back from source space.
        transform_params : dict
            Contains rotation, scale, translation, centroids, and RMS error.
        """
        def to_nx3(vec):
            if vec is None:
                return None
            a = np.asarray(vec, dtype=float)
            if a.ndim == 1:
                if a.size % 3 != 0:
                    raise ValueError("Point arrays must be length multiple of 3.")
                return np.column_stack([a[0::3], a[1::3], a[2::3]])
            elif a.ndim == 2 and a.shape[1] == 3:
                return a
            else:
                raise ValueError("Points must be (N,3) or 1D flattened with length multiple of 3.")

        X = to_nx3(target_points)   # target
        Y = to_nx3(source_points)   # source
        P = to_nx3(extra_points)    # extra

        n = X.shape[0]
        centroid_X = X.mean(axis=0)
        centroid_Y = Y.mean(axis=0)

        Xc = X - centroid_X
        Yc = Y - centroid_Y

        # covariance
        cov = (Yc.T @ Xc) / n

        # SVD
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            if not allow_reflection:
                S[2, 2] = -1.0

        R = U @ S @ Vt
        scale = np.sum(D * np.diag(S)) / (np.sum(Xc ** 2) / n)
        t = centroid_Y - scale * (centroid_X @ R.T)

        # forward: target → source
        def forward(pts):
            return scale * (pts @ R.T) + t

        # reverse: source → target
        def reverse(pts):
            return ((pts - t) @ R) / scale

        aligned_target = forward(X)
        aligned_extra = forward(P) if P is not None else None

        restored_target = reverse(Y)
        restored_extra = reverse(P) if P is not None else None

        rms = np.sqrt(np.mean(np.sum((aligned_target - Y) ** 2, axis=1)))

        transform_params = {
            "rotation_matrix": R,
            "scale": float(scale),
            "translation": tuple(t.tolist()),
            "centroid_target": centroid_X,
            "centroid_source": centroid_Y,
            "rms_error": float(rms),
        }

        return aligned_target, aligned_extra, restored_target, restored_extra, transform_params

    # Requires Jaxdifferentalgeometry package

    from jaxgeometry.Riemannian import metric
    from jaxgeometry.dynamics import Hamiltonian
    from jaxgeometry.Riemannian import Log
    from jaxgeometry.dynamics import flow_differential
    """
    Estimate diffeomorphism transformation from source outline to target outline using JAXGeometry,
    then apply it to surface points.
    
    Args:
      source_outline_points: numpy array of shape (n, 3) with source outline landmark positions
      target_outline_points: numpy array of shape (n, 3) with target outline landmark positions
      surface_points: numpy array of shape (m, 3) with surface points to transform
      
    Returns:
      Transformed surface points as numpy array
    """
    print(np.shape(source_outline_points))
    
    # reverse the order of target_outline_points, note it is in 3d coordinates, such that the last point because the first and ifrst point because the last
    #output = np.concatenate([source_outline_points,target_outline_points], axis=0)
   # return np.array(output)
    # Step 1: Align target landmarks to source landmarks
    print("Step 1: Aligning target landmarks to source landmarks")
    aligned_target_points, transform_params = align_shapes_3d(
        target_outline_points, 
        source_outline_points
    )
    aligned_target_points = aligned_target_points.reshape(-1, 3)
    print(f"Aligned target shape: {aligned_target_points.shape}")
    
    # Convert numpy arrays to JAX arrays using aligned target
    source_jax = jnp.array(source_outline_points.flatten())
    target_jax = jnp.array(aligned_target_points.flatten())
   # output = np.concatenate([source_jax.reshape(-1, 3),target_jax.reshape(-1, 3)], axis=0)
   # return np.array(aligned_target_points)

    print(f"Estimating diffeomorphism from {len(source_outline_points)} outline point pairs")
    print(f"Source outline shape: {source_jax.shape}")
    print(f"Target outline shape: {target_jax.shape}")
    
    d = 3
    n_landmarks = jnp.shape(source_jax)[0]//d
    
    # Calculate average distance between all landmarks as sigma_k
    source_reshaped = source_jax.reshape(n_landmarks, d)
    pairwise_dists = jnp.linalg.norm(source_reshaped[:, None] - source_reshaped[None, :], axis=2)
    mask = jnp.triu(jnp.ones((n_landmarks, n_landmarks)), k=1)
    avg_distance = jnp.sum(pairwise_dists * mask) / jnp.sum(mask)
    print("\t\t\t\t\t\t\t\t\tThe avverage distasnce is:", avg_distance)
    sigma_k = jnp.array([avg_distance]*d)

    M = landmarks(n_landmarks,k_sigma=sigma_k*jnp.eye(d),m=d) 
    # Riemannian structure

    metric.initialize(M)
    q = M.coords(jnp.array(source_jax))
    v =  (jnp.array(target_jax),[0])
    Hamiltonian.initialize(M)
    # Logarithm map
    Log.initialize(M,f=M.Exp_Hamiltonian)

    # Estimate momentum 
    p = M.Log(q,v)[0]

    print(f"Applying diffeomorphism to {len(surface_points)} surface points")
    
    # Convert surface points to coordinate format expected by manifold
    surface_jax = jnp.array(surface_points.flatten())
    q_surface = M.coords(surface_jax)
    print(surface_jax.shape)
    # Apply Hamiltonian dynamics to get the transformation

    def ode_Hamiltonian_advect(c,y):
        t,x,chart = c
        qp, = y
        q = qp[0]
        p = qp[1]

        dxt = jnp.tensordot(M.K(x,q),p,(1,0)).reshape((-1,M.m))
        return dxt
    M.Hamiltonian_advect = lambda xs,qps,dts: integrate(ode_Hamiltonian_advect,
                                                        None,
                                                        xs[0].reshape((-1,M.m)),
                                                        xs[1],
                                                        dts,
                                                        qps[::1])

    (_, qps, _) = M.Hamiltonian_dynamics(q, p, dts(n_steps=100))
    
    _,xs = M.Hamiltonian_advect(q_surface,qps, dts(n_steps=100))

    print("Shape of xs")
    print(xs.shape)
    # Get final transformed positions for all points at the last time step
    transformed_surface = xs[-1, :, :].reshape(-1, 3)  # Take final time step, all points
    
    # Step 2: Apply reverse alignment to get back to original coordinate system
    print("Step 2: Applying reverse alignment to surface points")
    # Apply the reverse transformation using the stored parameters
    aligned_target, aligned_extra, restored_target, restored_extra, params = \
        align_shapes_3d_with_followers(target_outline_points, source_outline_points, extra_points=transformed_surface)

   # final_surface_points = aligned_target.reshape(-1, 3)
    #print(f"Final transformation completed. Output shape: {final_surface_points.shape}")
   
    return np.array(restored_extra)


  def run(self, meshSourceNode, meshTargetNode, LMSourceNode, LMTargetNode, sourceOutline, targetOutline, placementValue, usePointCount=True, penaltyWeights=None):


    semiLandmarks = self.applyPatch(meshSourceNode, meshTargetNode, LMSourceNode, LMTargetNode, sourceOutline, targetOutline, placementValue, usePointCount, penaltyWeights)


    return True


  def applyPatch(self, meshSourceNode, meshTargetNode, LMSourceNode, LMTargetNode, sourceOutline, targetOutline, placementValue, usePointCount=True, penaltyWeights=None):
    
    if usePointCount:
        print(f"Source Outline: {sourceOutline}, Target Outline: {targetOutline}, Point count: {placementValue}")
    else:
        print(f"Source Outline: {sourceOutline}, Target Outline: {targetOutline}, Distance spacing: {placementValue}")
    
    import numpy as np
    import vtk
    import slicer
  #  from help_functions import generate_surface_landmarks_from_outline
    source_surface_polydata = meshSourceNode.GetPolyData()
    target_surface_polydata = meshTargetNode.GetPolyData()

    # ==================== MAIN PROCESSING ====================
    
    # Extract existing landmark points from source directly (no interpolation)
    source_outline_positions = []
    max_source_landmarks = LMSourceNode.GetNumberOfControlPoints()
    
    for lm_idx in sourceOutline:
        idx = int(lm_idx - 1)  # Convert to 0-based index
        if idx < 0 or idx >= max_source_landmarks:
            print(f"Warning: Source landmark index {lm_idx} is out of range (1-{max_source_landmarks}). Skipping.")
            continue
        
        point = LMSourceNode.GetNthControlPointPosition(idx)
        if point is None:
            print(f"Warning: Could not get position for source landmark {lm_idx}. Skipping.")
            continue
        
        source_outline_positions.append(list(point))
    
    if len(source_outline_positions) < 2:
        print(f"Error: Need at least 2 valid source outline landmark points, got {len(source_outline_positions)}")
        return None
    
    source_outline_positions = np.array(source_outline_positions)
    print(f"Using {len(source_outline_positions)} existing source outline landmark points as boundary")
    
    # Extract existing landmark points from target directly (no interpolation)
    target_outline_positions = []
    max_target_landmarks = LMTargetNode.GetNumberOfControlPoints()
    
    for lm_idx in targetOutline:
        idx = int(lm_idx - 1)  # Convert to 0-based index
        if idx < 0 or idx >= max_target_landmarks:
            print(f"Warning: Target landmark index {lm_idx} is out of range (1-{max_target_landmarks}). Skipping.")
            continue
        
        point = LMTargetNode.GetNthControlPointPosition(idx)
        if point is None:
            print(f"Warning: Could not get position for target landmark {lm_idx}. Skipping.")
            continue
        
        target_outline_positions.append(list(point))
    
    if len(target_outline_positions) < 2:
        print(f"Error: Need at least 2 valid target outline landmark points, got {len(target_outline_positions)}")
        return None
    
    target_outline_positions = np.array(target_outline_positions)
    print(f"Using {len(target_outline_positions)} existing target outline landmark points as boundary")
    
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
        point_locator.SetDataSet(source_surface_polydata)
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
                surface_point = source_surface_polydata.GetPoint(closest_id)
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

    # Create 3D surface from source outline and sample on it
    #num_surface_points = int(placementValue) if usePointCount else 50  # Use placement value for point count, default for distance method
    #source_surface_result = create_3d_surface_from_outline(source_outline_positions, source_surface_polydata, num_surface_points)
    
    source_surface_points = generate_surface_landmarks_from_outline(source_outline_positions, placementValue, 30, usePointCount)


  
    
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
                    mean_nearest3 = np.mean(nearest3_distances)
                    # Penalize points whose 3-nearest neighbor distances deviate from the mean
                    spacing_penalty = np.sum((nearest3_distances - mean_nearest3) ** 2)
            
            # Term 4: Distance to boundary points (maintain boundary awareness)
            boundary_penalty = 0.0
            if source_outline_positions is not None and len(source_outline_positions) > 0:
                for pos in positions:
                    boundary_distances = np.linalg.norm(source_outline_positions - pos, axis=1)
                    min_boundary_dist = np.min(boundary_distances)
                    # Encourage some minimum distance from boundary
                    if min_boundary_dist < 1.0:  # Adjust threshold as needed
                        boundary_penalty += (1.0 - min_boundary_dist)**2
            
         
            # Combine penalties
            total_loss = (
                penalty_weights.get('original_distance', 1) * original_distance_penalty +  # Stay near original
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
    
    # Create semilandmark node for 3D surface sampled points from source

    # Optimize source surface points to lie exactly on the source mesh surface
   # optimized_source_surface_points = optimize_points_to_surface(
            #source_surface_points, 
            #source_surface_polydata, 
            #penaltyWeights )
    
    optimized_source_surface_points = source_surface_points
    source_surface_node_name = f"semiLM_3D_Source_Surface_Optimized_{len(optimized_source_surface_points)}pts"
    source_surface_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", source_surface_node_name)
    
    # Add optimized source surface-sampled points
    for i, surface_point in enumerate(optimized_source_surface_points):
        label = f"source_surface_opt_{i+1}"
        source_surface_node.AddControlPoint(surface_point, label)
    
    # Set display properties
    import random  # Ensure we use Python's standard random module
    source_surface_node.SetLocked(True)
    source_surface_display_node = source_surface_node.GetDisplayNode()
    source_random_color = [random.random(), random.random(), random.random()]
    source_surface_display_node.SetColor(*source_random_color)
    # Set a slightly darker random color for selected color
    source_selected_color = [max(0, c - 0.2) for c in source_random_color]
    source_surface_display_node.SetSelectedColor(*source_selected_color)
    source_surface_display_node.PointLabelsVisibilityOff()
    
    print(f"Created {len(optimized_source_surface_points)} optimized 3D source surface points (projected to source mesh surface)")
    
    # Print source surface information

    
    # ==================== DIFFEOMORPHISM-BASED TRANSFORMATION ====================
    
    # Step 1: Estimate diffeomorphism from outline points and apply to surface points
    print("\n=== ESTIMATING DIFFEOMORPHISM FROM OUTLINE POINTS ===")
    transformed_surface_points = self.estimate_and_apply(source_outline_positions, target_outline_positions, optimized_source_surface_points)
  
    # Step 3: Project transformed points to target surface for refinement
    target_projected_points = []
    
    # Create VTK point locator for target surface projection
    target_point_locator = vtk.vtkPointLocator()
    target_point_locator.SetDataSet(target_surface_polydata)
    target_point_locator.BuildLocator()
    
    for transformed_point in transformed_surface_points:
        closest_id = target_point_locator.FindClosestPoint(transformed_point)
        target_surface_point = target_surface_polydata.GetPoint(closest_id)
        target_projected_points.append(list(target_surface_point))
    
    # Create target semilandmark node
    target_surface_node_name = f"semiLM_3D_Target_Surface_Diffeomorphism_{len(target_projected_points)}pts"
    target_surface_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", target_surface_node_name)
    
    # Add projected target surface points
    #for i, target_point in enumerate(target_projected_points):
    for i, target_point in enumerate(transformed_surface_points):
        label = f"target_diffeo_{i+1}"
        target_surface_node.AddControlPoint(target_point, label)
    
    # Set display properties for target
    target_surface_node.SetLocked(True)
    target_surface_display_node = target_surface_node.GetDisplayNode()
    target_random_color = [random.random(), random.random(), random.random()]
    target_surface_display_node.SetColor(*target_random_color)
    target_selected_color = [max(0, c - 0.2) for c in target_random_color]
    target_surface_display_node.SetSelectedColor(*target_selected_color)
    #target_surface_display_node.PointLabelsVisibilityOff()
    
    print(f"Created {len(target_projected_points)} transformed 3D target surface points using diffeomorphism transformation")
    
    return source_surface_node, target_surface_node


