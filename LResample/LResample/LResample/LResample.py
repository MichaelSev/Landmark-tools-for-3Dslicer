import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import fnmatch
import  numpy as np
import random
import math
import jax
import jax.numpy as jnp
from jax import grad, jit




#
# CreateSemiLMPatches
#

class LResample(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "LResample" # TODO make this more human readable by adding spaces
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

class LResampleWidget(ScriptedLoadableModuleWidget):
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
    self.outlinePointsInput1.setText("1,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,4")  # Set initial value to empty string
  


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
    self.originalDistancePenalty.singleStep = 0.1
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
    self.surfaceDistancePenalty.value = 1.0
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
    self.smoothnessPenalty.singleStep = 0.1
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
      placement_value = int(self.gridSamplingRate.value) 
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
  
  # Optimization constants
  OPTIMIZATION_MAX_ITER = 10
  OPTIMIZATION_FTOL = 1e-12
  OPTIMIZATION_GTOL = 1e-8
  OPTIMIZATION_EPS = 1e-8
  PERTURBATION_THRESHOLD = 1.0
  PERTURBATION_STD = 0.1
  
  # Display colors
  SEMILANDMARK_COLOR = (0.0, 1.0, 0.0)
  SEMILANDMARK_SELECTED_COLOR = (0.0, 0.8, 0.0)

  def interpolate_points_by_method(self, landmark_points, placement_value, use_point_count=True):
    """Interpolate points evenly along the landmark path by point count or distance."""
    
    # Validate input points
    for i, pt in enumerate(landmark_points):
        if pt is None:
            raise ValueError(f"Landmark point at index {i} is None")
    
    # Calculate cumulative distances along the path
    cumulative_distances = [0.0]
    total_distance = 0.0
    
    for i in range(len(landmark_points) - 1):
        pt1 = np.array(landmark_points[i])
        pt2 = np.array(landmark_points[i + 1])
        segment_distance = np.linalg.norm(pt2 - pt1)
        total_distance += segment_distance
        cumulative_distances.append(total_distance)
    
    # Determine target distances based on method
    if use_point_count:
        # Original method: specific number of points
        sample_rate = int(placement_value)
        target_distances = np.linspace(0, total_distance, sample_rate)
    else:
        # New method: specific distance spacing
        distance_spacing = placement_value
        num_points = int(total_distance / distance_spacing) + 1
        target_distances = np.linspace(0, total_distance, num_points)
    
    interpolated_points = []
    
    for target_dist in target_distances:
        # Find which segment this target distance falls into
        segment_idx = 0
        for i in range(len(cumulative_distances) - 1):
            if cumulative_distances[i] <= target_dist <= cumulative_distances[i + 1]:
                segment_idx = i
                break
        
        # Handle edge case: if target_dist is exactly the total distance
        if target_dist >= total_distance:
            interpolated_points.append(np.array(landmark_points[-1]))
            continue
        
        # Interpolate within the found segment
        segment_start_dist = cumulative_distances[segment_idx]
        segment_end_dist = cumulative_distances[segment_idx + 1]
        segment_length = segment_end_dist - segment_start_dist
        
        if segment_length > 0:
            # Calculate interpolation parameter (0 to 1)
            t = (target_dist - segment_start_dist) / segment_length
            
            # Linear interpolation between segment endpoints
            start_point = np.array(landmark_points[segment_idx])
            end_point = np.array(landmark_points[segment_idx + 1])
            interpolated_point = start_point + t * (end_point - start_point)
            
            interpolated_points.append(interpolated_point)
        else:
            # Zero-length segment, just use the start point
            interpolated_points.append(np.array(landmark_points[segment_idx]))
    
    
    return interpolated_points

  def validate_and_extract_landmarks(self, landmark_node, outline_indices):
    """Extract and validate landmark points with comprehensive error checking."""
    landmark_points = []
    max_landmarks = landmark_node.GetNumberOfControlPoints()
    
    for landmark_idx in outline_indices:
        zero_based_idx = int(landmark_idx - 1)  # Convert to 0-based index
        if zero_based_idx < 0 or zero_based_idx >= max_landmarks:
            print(f"Warning: Landmark index {landmark_idx} is out of range (1-{max_landmarks}). Skipping.")
            continue
        
        point = landmark_node.GetNthControlPointPosition(zero_based_idx)
        if point is None:
            print(f"Warning: Could not get position for landmark {landmark_idx}. Skipping.")
            continue
        
        landmark_points.append(point)
    
    if len(landmark_points) < 2:
        raise ValueError(f"Need at least 2 valid landmark points, got {len(landmark_points)}")
    
    return landmark_points



  @staticmethod
  def jax_loss_function(positions_flat, first_landmark, last_landmark, intermediate_original, 
                       surface_points_array, penalty_weights):
    """JAX-compatible loss function for automatic differentiation."""
    # Reshape intermediate positions
    num_intermediate = len(intermediate_original)
    intermediate_opt = positions_flat.reshape(num_intermediate, 3)
    
    # Reconstruct full positions with fixed endpoints
    full_positions = jnp.vstack([
        first_landmark.reshape(1, -1),
        intermediate_opt,
        last_landmark.reshape(1, -1)
    ])
    
    # Term 1: Stay close to original positions
    original_distance_penalty = jnp.sum((intermediate_opt - intermediate_original)**2)
    
    # Term 2: Distance to surface penalty (simplified for JAX)
    # Use approximate surface distance with pre-computed surface points
    distances_matrix = jnp.linalg.norm(
        full_positions[:, None, :] - surface_points_array[None, :, :], axis=2
    )
    min_distances = jnp.min(distances_matrix, axis=1)
    surface_distance_penalty = jnp.sum(min_distances**2)
    
    # Term 3: Equal spacing penalty (vectorized)
    spacing_penalty = 0.0
    if full_positions.shape[0] >= 3:
        all_distances = jnp.linalg.norm(jnp.diff(full_positions, axis=0), axis=1)
        expected_distance = jnp.mean(all_distances)
        
        if len(all_distances) > 1:
            prev_distances = all_distances[:-1]
            next_distances = all_distances[1:]
            spacing_penalty = jnp.sum((prev_distances - expected_distance)**2) + \
                            jnp.sum((next_distances - expected_distance)**2)
    
    # Term 4: Smoothness penalty (vectorized)
    smoothness_penalty = 0.0
    if full_positions.shape[0] >= 3:
        second_derivatives = full_positions[:-2] - 2*full_positions[1:-1] + full_positions[2:]
        smoothness_penalty = jnp.sum(second_derivatives**2)
    
    # Total loss
    total_loss = (
        penalty_weights['original_distance'] * original_distance_penalty +
        penalty_weights['surface_distance'] * surface_distance_penalty +
        penalty_weights['spacing'] * spacing_penalty +
        penalty_weights['smoothness'] * smoothness_penalty
    )
    
    return total_loss

  def run(self, meshNode, LMNode, outline, placementValue, usePointCount=True, penaltyWeights=None):

 
    semiLandmarks = self.applyPatch(meshNode, LMNode, outline, placementValue, usePointCount, penaltyWeights)
    return True



  def applyPatch(self, meshNode, LMNode, outline, placementValue, usePointCount=True, penaltyWeights=None):
    """Apply patch optimization using extracted and optimized methods."""
    
    if usePointCount:
        print(f"Outline: {outline}, Point count: {placementValue}")
    else:
        print(f"Outline: {outline}, Distance spacing: {placementValue}")
    
    surface_polydata = meshNode.GetPolyData()
    
    # Create single VTK point locator for reuse throughout function
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(surface_polydata)
    point_locator.BuildLocator()
    
    def optimize_with_jax(initial_positions, original_positions):
        """JAX-based optimization using automatic differentiation."""
        
        # Store the first and last landmark positions (fixed endpoints)
        first_landmark = original_positions[0].copy()
        last_landmark = original_positions[-1].copy()
        
        # Only optimize the intermediate points (exclude first and last)
        if len(initial_positions) <= 2:
            return initial_positions
        
        intermediate_positions = initial_positions[1:-1]
        intermediate_original = original_positions[1:-1]
        
        print("Using JAX optimization with automatic differentiation")
        # Since we're in a nested function, directly implement the JAX optimization here
        # Pre-compute surface points for JAX optimization
        n_surface_points = min(2000, surface_polydata.GetNumberOfPoints())
        surface_points_indices = np.random.choice(surface_polydata.GetNumberOfPoints(), 
                                                n_surface_points, replace=False)
        surface_points_array = np.array([
            surface_polydata.GetPoint(idx) for idx in surface_points_indices
        ])
        
        # Convert to JAX arrays
        first_landmark_jax = jnp.array(first_landmark)
        last_landmark_jax = jnp.array(last_landmark)
        intermediate_original_jax = jnp.array(intermediate_original)
        surface_points_jax = jnp.array(surface_points_array)
        penalty_weights_jax = {k: float(v) for k, v in penaltyWeights.items()}
        
        # Create JIT-compiled loss and gradient functions
        @jit
        def loss_fn(positions_flat):
            return CreateSemiLMPatchesLogic.jax_loss_function(positions_flat, first_landmark_jax, last_landmark_jax,
                                        intermediate_original_jax, surface_points_jax, penalty_weights_jax)
        
        grad_fn = jit(grad(loss_fn))
        
        # Initialize
        positions_flat = jnp.array(intermediate_positions.flatten())
        learning_rate = 0.01
        
        # Simple gradient descent with momentum
        momentum = jnp.zeros_like(positions_flat)
        momentum_decay = 0.9
        
        for iteration in range(100):  # Fixed number of iterations
            gradients = grad_fn(positions_flat)
            
            # Update with momentum
            momentum = momentum_decay * momentum - learning_rate * gradients
            positions_flat = positions_flat + momentum
            
            # Optional: adaptive learning rate
            if iteration % 50 == 0:
                loss_val = loss_fn(positions_flat)
                print(f"JAX Iteration {iteration}: Loss = {loss_val}")
                
            # Simple convergence check
            if jnp.linalg.norm(gradients) < 1e-6:
                break
        
        # Reconstruct final positions
        optimized_intermediate = np.array(positions_flat.reshape(-1, 3))
        final_positions = np.vstack([
            first_landmark.reshape(1, -1),
            optimized_intermediate,
            last_landmark.reshape(1, -1)
        ])
        
        return final_positions


    
    # Main processing using extracted methods
    try:
        landmark_points = self.validate_and_extract_landmarks(LMNode, outline)
        initial_interpolated = self.interpolate_points_by_method(landmark_points, placementValue, usePointCount)
        
        if not initial_interpolated:
            print("Error: No interpolated points generated")
            return None
        
        # Optimize array operations - direct numpy array creation
        initial_positions = np.array(initial_interpolated)
        original_positions = initial_positions.copy()
        
        # Optimize positions using JAX
        optimized_positions = optimize_with_jax(initial_positions, original_positions)
        
        # Create semilandmark node with optimized naming
        node_name = f"semiLM_OPT_outline_{len(optimized_positions)}pts"
        semilandmark_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
        
        # Add final optimized points
        for point_idx, final_point in enumerate(optimized_positions):
            label = f"opt_{point_idx+1}"
            semilandmark_node.AddControlPoint(final_point, label)
        
        # Finalize display with constants
        semilandmark_node.SetLocked(True)
        display_node = semilandmark_node.GetDisplayNode()
        display_node.SetColor(*self.SEMILANDMARK_COLOR)
        display_node.SetSelectedColor(*self.SEMILANDMARK_SELECTED_COLOR)
        display_node.PointLabelsVisibilityOff()
        
        return semilandmark_node
        
    except ValueError as error:
        print(f"Error in applyPatch: {error}")
        return None



