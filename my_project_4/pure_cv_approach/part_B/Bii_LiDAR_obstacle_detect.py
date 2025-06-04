import os
import cv2
import numpy as np
from time import time
from sklearn.cluster import DBSCAN

from Bi_road_detection_pcd import (
    detect_ground_plane,
    project_lidar_to_image,
    load_velodyne_bin,
    load_calibration,
    filter_visible_points,
    filter_by_height
)

# --- Ρύθμιση παραμέτρων για ευκολία...
# ΝΑΙ, είναι GLOBAL... ΝΑΙ, δεν θέλω... αλλά τι να κάνεις τώρα!!!
CONFIG = {
    'ground_removal': {
        'distance_threshold': 0.02,
        'num_iterations':    10000,
        'ground_max_dist':    0.15,
        'non_ground_max_dist': 0.3
    },
    'height_filter': {
        'min_height': 0.3,
        'max_height': 2.5
    },
    'clustering': {
        'eps':        0.3,
        'min_samples': 10
    },
    'projection': {
        'min_box_area': 30
    },
    'visualization': {
        'dot_radius':          4,
        'dot_thickness':      -1, # filled κύκλοι
        'max_lidar_distance': 50.
    }
}

# Χρώματα βάση απόσταση από το LiDAR
PROXIMITY_COLORS = [
    (0, 0, 255  ), # Κόκκινο  / πολύ κοντά!
    (0, 128, 255), # Πορτοκαλί/ κοντά
    (0, 255, 255), # Κίτρινο  / μέτρια απόσταση
    (0, 255, 0  ), # Πράσινο  / μακριά...
]

def filter_points_near_plane(points:      np.ndarray,
                             plane_model: np.ndarray,
                             max_dist:    float = 0.15,
                             invert:      bool = False) -> np.ndarray:
    ''' Φιλτράρισμα σημείων βάσει απόστασης από επίπεδο. '''
    (a, b, c, d) = plane_model
    normal       = np.array([a, b, c])
    distances    = np.abs(points @ normal + d) / np.linalg.norm(normal)
    
    return points[distances > max_dist] \
        if invert else points[distances <= max_dist];

def ransac_ground_removal(points: np.ndarray) -> tuple:
    '''
    Χρησιμοποιώντας RANSAC, τα σημεία
    χωρίζονται σε έδαφος/δρόμος και μη!
    '''
    temp = CONFIG['ground_removal']
    
    (_, ground_plane) = detect_ground_plane(
        points,
        distance_threshold = temp['distance_threshold'],
        num_iterations     = temp['num_iterations']
    )
    
    non_ground = filter_points_near_plane(
        points,
        ground_plane, 
        max_dist = temp['non_ground_max_dist'], 
        invert   = True
    )
    ground = filter_points_near_plane(
        points,
        ground_plane, 
        max_dist = temp['ground_max_dist']
    )
    
    return (non_ground, ground, ground_plane);

def cluster_obstacles_dbscan(points: np.ndarray) -> np.ndarray:
    ''' DBASCAN για ομαδοποίηση σημείων. '''
    temp       = CONFIG['clustering']
    clustering = DBSCAN(
        eps         = temp['eps'],
        min_samples = temp['min_samples']
    ).fit(points)

    return clustering.labels_;

# --- Cluster Helpers ---
def group_clusters(points: np.ndarray,
                   labels: np.ndarray) -> dict:
    ''' Ομαδοποίηση σημείων σε clusters βάσει labels. '''
    clusters = {}
    for (pt, lbl) in zip(points, labels):
        if lbl == -1: # Δεν μας νοιάζει ο θόρυβος!
            continue;
        clusters.setdefault(lbl, []).append(pt)

    return {k: np.array(v) for (k, v) in clusters.items()};

def calculate_lidar_distance(cluster_points: np.ndarray) -> float:
    ''' Υπολογισμός απόστασης μεταξύ cluster και σημείου αναφοράς. '''
    # Θεωρούμε ότι το LiDAR [σημείο αναφοράς] είναι το (0, 0, 0)
    cluster_center = np.mean(cluster_points, axis = 0)
    
    # Υπολογισμός απόστασης μεταξύ σ.α. και cluster center 
    distance = np.linalg.norm(cluster_center)
    
    return distance;

def project_clusters_with_dots(clusters, ground_plane, Tr_velo_to_cam, P2, image_shape):
    """Project clusters to image and create colored dots based on LiDAR distance."""
    h, w = image_shape[:2]
    colored_dots = []
    max_lidar_distance = CONFIG['visualization']['max_lidar_distance']

    for cluster_id, cluster_points in clusters.items():
        # Calculate distance from LiDAR sensor
        lidar_distance = calculate_lidar_distance(cluster_points)
        
        # Only process obstacles within reasonable LiDAR range
        if lidar_distance > max_lidar_distance:
            continue
        
        # Project cluster to image
        _, u, v, _ = project_lidar_to_image(cluster_points, Tr_velo_to_cam, P2)
        uvs = np.stack([u, v], axis=1)

        # Keep only in-image projections
        valid_mask = ((uvs[:, 0] >= 0) & (uvs[:, 0] < w) & 
                     (uvs[:, 1] >= 0) & (uvs[:, 1] < h))
        uvs_valid = uvs[valid_mask]
        
        if len(uvs_valid) < 5:  # Need enough points for reliable detection
            continue

        # Get color based on LiDAR distance
        color = get_distance_color(lidar_distance)
        
        # Sample points for dot visualization
        num_dots = min(len(uvs_valid), max(8, len(uvs_valid) // 2))
        indices = np.linspace(0, len(uvs_valid) - 1, num_dots, dtype=int)
        sampled_uvs = uvs_valid[indices]
        
        for u_coord, v_coord in sampled_uvs:
            colored_dots.append({
                'center': (int(u_coord), int(v_coord)),
                'color': color,
                'distance': lidar_distance,
                'cluster_id': cluster_id
            })

    return [], colored_dots  # Return empty list for boxes

def get_distance_color(distance: float) -> tuple:
    ''' Επιστρέφει χρώμα βάσει απόστασης από το LiDAR.'''
    temp = PROXIMITY_COLORS[3] # Μακριά από LiDAR - Πράσινο

    if distance <= 5.:    # Πολύ κοντά στο LiDAR
        temp = PROXIMITY_COLORS[0] # Κόκκινο
    elif distance <= 10.: # Κοντά στο LiDAR
        temp = PROXIMITY_COLORS[1] # Πορτοκαλί
    elif distance <= 20.: # Μέτρια απόσταση από LiDAR
        temp = PROXIMITY_COLORS[2] # Κίτρινο
    
    return temp;

def draw_legend(image: np.ndarray) -> None:
    ''' Πινακάκι πληροφοριών χρώμα <-> απόσταση. '''
    (legend_x, legend_y) = (10, 10)
    legend_spacing = 25
    
    legend_items = [
        ('Very Close (0 - 5m)', PROXIMITY_COLORS[0]),
        ('Close      (5 -10m)', PROXIMITY_COLORS[1]),
        ('Moderate  (10-20m)',  PROXIMITY_COLORS[2]),
        ('Far        ( >20m )', PROXIMITY_COLORS[3]),
    ]
    
    # Legend background
    legend_bg = np.zeros(
        (len(legend_items) * legend_spacing + 10, 200, 3), dtype = np.uint8
    )
    legend_bg.fill(50) # Dark gray background
    
    for (i, (text, color)) in enumerate(legend_items):
        y_pos = legend_y + i * legend_spacing
        cv2.circle(legend_bg, (15, y_pos + 10), 5, color, -1)
        cv2.putText(
            legend_bg, text, (25, y_pos + 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
    
    # Overlay legend on image
    (h_legend, w_legend) = legend_bg.shape[:2]
    if (h_legend < image.shape[0]) and (w_legend < image.shape[1]):
        alpha = 0.7
        image[
            legend_y:legend_y + h_legend,
            legend_x:legend_x + w_legend
        ] = cv2.addWeighted(
                image[
                    legend_y:legend_y + h_legend,
                    legend_x:legend_x + w_legend
                ], 
                1 - alpha, legend_bg, alpha, 0
            )
    
    return;

def visualize_results(image:        np.ndarray,
                      colored_dots: list) -> np.ndarray:
    ''' Απεικόνιση των χρωματιστών dots στην εικόνα. '''
    temp = CONFIG['visualization']
    
    for dot in colored_dots:
        cv2.circle(
            image,
            dot['center'], temp['dot_radius'], 
            dot['color'], temp['dot_thickness']
        )
    
    draw_legend(image)
    
    return image;

def process_frame(image_path, bin_path, calib_path):
    '''Process a single frame and return results.'''
    # Load data
    image = cv2.imread(image_path)
    points = load_velodyne_bin(bin_path)
    Tr_velo_to_cam, P2 = load_calibration(calib_path)

    # Filter visible points
    visible_points = filter_visible_points(points, Tr_velo_to_cam, P2, image.shape)

    # Ground removal
    obstacles_raw, ground, ground_plane = ransac_ground_removal(visible_points)

    # Height filtering
    obstacles_filtered = filter_by_height(obstacles_raw, ground_plane,
                                          min_h=CONFIG['height_filter']['min_height'],
                                          max_h=CONFIG['height_filter']['max_height'])

    # Clustering
    labels = cluster_obstacles_dbscan(obstacles_filtered)
    clusters = group_clusters(obstacles_filtered, labels)

    # Project to image with colored dots (no boxes)
    boxes, colored_dots = project_clusters_with_dots(
        clusters, ground_plane, Tr_velo_to_cam, P2, image.shape)

    # Visualize results (dots only)
    result_image = visualize_results(image, colored_dots)

    return result_image, [], colored_dots, len(clusters)

def main():
    base_dir = os.path.dirname(__file__)
    
    for i in range(10, 50):
        name = f'um_0000{i:02d}'
        img_path = os.path.join(base_dir, '..', '..', 'KITTI', 'data_road', 'training', 'image_2', f'{name}.png')
        bin_path = os.path.join(base_dir, '..', '..', 'KITTI', 'data_road_velodyne', 'training', 'velodyne', f'{name}.bin')
        calib_path = os.path.join(base_dir, '..', '..', 'KITTI', 'data_road', 'training', 'calib', f'{name}.txt')

        if not all(os.path.exists(path) for path in [img_path, bin_path, calib_path]):
            print(f'⚠️  Missing files for: {name}')
            continue

        try:
            start_time = time()
            result_image, _, colored_dots, num_clusters = process_frame(
                img_path, bin_path, calib_path)
            processing_time = time() - start_time

            # Print frame statistics with LiDAR distance info
            if colored_dots:
                distances = [dot['distance'] for dot in colored_dots]
                avg_distance = np.mean(distances)
                min_distance = np.min(distances)
                max_distance = np.max(distances)
                print(f'✅ {name}: {num_clusters} clusters, {len(colored_dots)} obstacles detected '
                      f'(avg: {avg_distance:.1f}m, range: {min_distance:.1f}-{max_distance:.1f}m, {processing_time:.2f}s)')
            else:
                print(f'✅ {name}: {num_clusters} clusters, no obstacles in range ({processing_time:.2f}s)')

            # Display results
            cv2.imshow('Enhanced LiDAR Obstacle Detection', result_image)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Press 'q' to quit
                break
            elif key == ord('s'):  # Press 's' to save
                output_path = f'output_{name}.png'
                cv2.imwrite(output_path, result_image)
                print(f'💾 Saved: {output_path}')
                
        except Exception as e:
            print(f'❌ Error processing {name}: {str(e)}')
            continue

    cv2.destroyAllWindows()
    print('\n🏁 Processing complete!')

    return;

if __name__ == '__main__':
    main()
