    
import numpy as np
import vtk
import slicer


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
