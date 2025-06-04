import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

def region_growing_on_height_band(points, plane, min_h, max_h, 
                                 depth_radius=1.2, width_radius=0.6, 
                                 angle_thresh_deg=20.0, height_thresh=0.08,
                                 min_cluster_size=50, max_slope_deg=15.0):
    """
    Enhanced region growing for road detection with stricter width constraints
    to avoid including sidewalks while being more permissive in depth direction.
    
    Parameters:
    - depth_radius: Larger radius for connections along the depth (forward/backward)
    - width_radius: Smaller radius for connections along the width (left/right)  
    - angle_thresh_deg: Maximum angle difference between normals
    - height_thresh: Maximum height difference between neighboring points
    - min_cluster_size: Minimum points to consider a valid cluster
    - max_slope_deg: Maximum allowed slope for road surface
    """
    
    # Filter points by height band
    a, b, c, d = plane
    normal = np.array([a, b, c])
    dist = (points @ normal + d) / np.linalg.norm(normal)
    
    mask = (dist >= min_h) & (dist <= max_h)
    band_points = points[mask]
    
    if band_points.shape[0] < min_cluster_size:
        return np.empty((0, 3))
    
    # Create point cloud and compute normals with better parameters
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(band_points)
    
    # Use hybrid search with appropriate radius for normal estimation
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.5, max_nn=20
        )
    )
    normals = np.asarray(pcd.normals)
    
    # Filter out points with steep normals (likely walls, curbs, etc.)
    max_slope_rad = np.deg2rad(max_slope_deg)
    vertical_dot = np.abs(normals[:, 2])  # Z component (up direction)
    slope_mask = vertical_dot > np.cos(max_slope_rad)
    
    if np.sum(slope_mask) < min_cluster_size:
        # If too few points pass slope filter, relax the constraint
        slope_mask = vertical_dot > np.cos(np.deg2rad(25.0))  # More permissive fallback
    
    if np.sum(slope_mask) < 20:  # Still too few points
        slope_mask = np.ones(len(normals), dtype=bool)  # Use all points
    
    # Apply slope filter
    band_points = band_points[slope_mask]
    normals = normals[slope_mask]
    
    # Build adaptive KDTree for directional search
    tree = KDTree(band_points)
    
    # Initialize clustering variables
    visited = np.zeros(len(band_points), dtype=bool)
    labels = -np.ones(len(band_points), dtype=int)
    label = 0
    
    angle_thresh_rad = np.deg2rad(angle_thresh_deg)
    cos_thresh = np.cos(angle_thresh_rad)
    
    # Enhanced region growing with directional constraints
    for i in range(len(band_points)):
        if visited[i]:
            continue
            
        # Start new region
        queue = [i]
        visited[i] = True
        labels[i] = label
        cluster_points = [i]
        
        while queue:
            current_idx = queue.pop(0)
            current_point = band_points[current_idx]
            current_normal = normals[current_idx]
            
            # Use adaptive radius based on direction
            neighbors = _get_directional_neighbors(
                current_point, band_points, tree, 
                depth_radius, width_radius
            )
            
            for neighbor_idx in neighbors:
                if visited[neighbor_idx]:
                    continue
                
                neighbor_point = band_points[neighbor_idx]
                neighbor_normal = normals[neighbor_idx]
                
                # Check multiple criteria for road continuity
                if _is_valid_road_connection(
                    current_point, neighbor_point,
                    current_normal, neighbor_normal,
                    cos_thresh, height_thresh
                ):
                    visited[neighbor_idx] = True
                    labels[neighbor_idx] = label
                    queue.append(neighbor_idx)
                    cluster_points.append(neighbor_idx)
        
        # Only keep cluster if it meets size requirements
        if len(cluster_points) >= min_cluster_size:
            label += 1
        else:
            # Remove small cluster
            for idx in cluster_points:
                labels[idx] = -1
    
    # Return largest valid cluster
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return np.empty((0, 3))
    
    unique, counts = np.unique(valid_labels, return_counts=True)
    largest_label = unique[np.argmax(counts)]
    
    final_points = band_points[labels == largest_label]
    
    # Final refinement: remove outliers
    final_points = _remove_outliers(final_points, plane)
    
    return final_points


def _get_directional_neighbors(point, all_points, tree, depth_radius, width_radius):
    """
    Get neighbors with different radius constraints for depth vs width directions.
    Assumes depth is along X-axis and width is along Y-axis.
    """
    # Get all neighbors within the larger radius
    all_neighbors = tree.query_radius([point], r=depth_radius)[0]
    
    if len(all_neighbors) <= 1:
        return []
    
    # Calculate relative positions
    neighbor_points = all_points[all_neighbors]
    relative_pos = neighbor_points - point
    
    # Separate depth (X) and width (Y) components
    depth_dist = np.abs(relative_pos[:, 0])
    width_dist = np.abs(relative_pos[:, 1])
    
    # Apply different thresholds
    valid_mask = (depth_dist <= depth_radius) & (width_dist <= width_radius)
    
    return all_neighbors[valid_mask]


def _is_valid_road_connection(p1, p2, n1, n2, cos_thresh, height_thresh):
    """
    Check if two points should be connected in the road surface.
    """
    # Normal similarity check
    if np.dot(n1, n2) < cos_thresh:
        return False
    
    # Height difference check
    height_diff = abs(p1[2] - p2[2])
    if height_diff > height_thresh:
        return False
    
    # Distance-based height tolerance (closer points can have less height difference)
    distance = np.linalg.norm(p1 - p2)
    adaptive_height_thresh = min(height_thresh, distance * 0.1)
    
    if height_diff > adaptive_height_thresh:
        return False
    
    # Gradient check (road shouldn't be too steep)
    if distance > 0.1:  # Avoid division by zero
        gradient = height_diff / distance
        if gradient > 0.15:  # 15% grade maximum
            return False
    
    return True


def _validate_road_cluster(cluster_points, plane):
    """
    Validate if a cluster represents a valid road segment.
    Made more permissive to avoid rejecting valid road segments.
    """
    if len(cluster_points) < 30:  # Reduced from 50
        return False
    
    # Check cluster compactness and shape - more permissive
    center = np.mean(cluster_points, axis=0)
    distances = np.linalg.norm(cluster_points - center, axis=1)
    
    # Road clusters shouldn't be too spread out - more permissive
    if np.std(distances) > 8.0:  # Increased from 5.0
        return False
    
    # Check height variation within cluster - more permissive
    a, b, c, d = plane
    normal = np.array([a, b, c])
    heights = (cluster_points @ normal + d) / np.linalg.norm(normal)
    
    if np.std(heights) > 0.12:  # Increased from 0.08
        return False
    
    return True


def _remove_outliers(points, plane, std_threshold=2.0):
    """
    Remove statistical outliers from the final road points.
    """
    if len(points) < 10:
        return points
    
    # Calculate distances from plane
    a, b, c, d = plane
    normal = np.array([a, b, c])
    distances = np.abs((points @ normal + d) / np.linalg.norm(normal))
    
    # Remove points that are too far from the plane
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + std_threshold * std_dist
    
    inlier_mask = distances <= threshold
    
    return points[inlier_mask]

def analyze_height_distribution(points: np.ndarray,
                                plane:  np.ndarray) -> None:
    '''
    Υπολογίζει histogram των αποστάσεων σημείων
    από το επίπεδο. Επιστρέφει τα ύψη ταξινομημένα.
    '''
    (a, b, c, d) = plane
    normal = np.array([a, b, c])
    height = (points @ normal + d) / np.linalg.norm(normal)

    # Ιστογράφημα
    (counts, bins) = np.histogram(height, bins = 200)
    max_bin_index = np.argmax(counts)
    mode_center = 0.5 * (bins[max_bin_index] + bins[max_bin_index + 1])

    return (np.sort(height), mode_center);

# Updated function call in your main code:
def enhanced_region_growing_road_detection(points, plane,
                                         vehicle_position=None):
    """
    Main function to call the enhanced region growing with optimized parameters.
    
    Parameters:
    - vehicle_position: If provided, helps prioritize road areas near the vehicle
    """
    # Slightly expand the height band around the mode
    height_tolerance = 0.06  # Increased from 0.04 for better coverage
    (_, mode_h) = analyze_height_distribution(points, plane)
    
    result = region_growing_on_height_band(
        points, plane,
        min_h=mode_h - height_tolerance,
        max_h=mode_h + height_tolerance,
        depth_radius=2.,       # More permissive in depth direction
        width_radius=0.4,      # More permissive in width direction  
        angle_thresh_deg=20.0, # More permissive normal similarity
        height_thresh=0.08,    # More permissive height difference
        min_cluster_size=50,   # Smaller minimum cluster size
        max_slope_deg=20.0     # More permissive slope constraint
    )
    
    return result