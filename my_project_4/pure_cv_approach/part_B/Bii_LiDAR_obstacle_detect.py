import os
import cv2
import numpy as np
from time import time
from sklearn.cluster import DBSCAN

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Αλλιώς, ModuleNotFoundError: No module named
# 'Bi_road_detection_pcd' όταν το Bii γίνεται import...
from Bi_road_detection_pcd import (
    my_road_from_pcd_is,
    filter_visible_points,
    project_lidar_to_image,
    load_velodyne_bin,
    load_calibration,
    filter_by_height,
    overlay_mask
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
        'dot_radius':            2,
        'dot_thickness':        -1, # filled κύκλοι
        'max_lidar_distance':  50.,
        'small_size_threshold': 20
    }
}

# Χρώματα βάση απόσταση από το LiDAR
PROXIMITY_COLORS = [
    (0, 0, 255  ), # Κόκκινο  / πολύ κοντά!
    (0, 128, 255), # Πορτοκαλί/ κοντά
    (0, 255, 255), # Κίτρινο  / μέτρια απόσταση
    (0, 255, 0  ), # Πράσινο  / μακριά...
]

def cluster_obstacles_dbscan(points: np.ndarray) -> np.ndarray:
    ''' DBSCAN για ομαδοποίηση σημείων. '''
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

def project_clusters_with_dots(clusters:       dict,
                               Tr_velo_to_cam: np.ndarray,
                               P2:             np.ndarray,
                               image_shape:    tuple) -> list:
    '''
    Προβολή clusters στην εικόνα και δημιουργία
    χρωματιστών dots βάσει απόστασης LiDAR.
    '''
    (h, w)       = image_shape[:2]
    colored_dots = []
    temp         = CONFIG['visualization']

    for (cluster_id, cluster_points) in clusters.items():
        lidar_distance = calculate_lidar_distance(cluster_points)
        
        # Λαμβάνουμε υπόψη μόνο clusters σε λογική απόσταση!
        if lidar_distance > temp['max_lidar_distance']:
            continue;
        
        (_, u, v, _) = project_lidar_to_image(
            cluster_points, Tr_velo_to_cam, P2
        )
        uvs = np.stack([u, v], axis = 1)

        # Κρατάμε μόνο τις προβολές που είναι εντός εικόνας!
        valid_mask = ((uvs[:, 0] >= 0) & (uvs[:, 0] < w) & 
                      (uvs[:, 1] >= 0) & (uvs[:, 1] < h))
        uvs_valid = uvs[valid_mask]
        
        # Αν το cluster είναι πολύ μικρό, το αγνοούμε...
        if len(uvs_valid) < temp['small_size_threshold']:
            continue;

        color = get_distance_color(lidar_distance)
        
        # Δειγματοληψία για dotted effect
        num_dots = min(
            len(uvs_valid),
            max(8, len(uvs_valid) // 2)
        )
        indices = np.linspace(
            0, len(uvs_valid) - 1, num_dots, dtype = int
        )
        sampled_uvs = uvs_valid[indices]
        
        for (u_coord, v_coord) in sampled_uvs:
            colored_dots.append({
                'center':     (int(u_coord), int(v_coord)),
                'color':      color,
                'distance':   lidar_distance,
                'cluster_id': cluster_id
            })

    return colored_dots;

def get_distance_color(distance: float) -> tuple:
    ''' Επιστρέφει χρώμα βάσει απόστασης από το LiDAR scanner. '''
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

# --- Helpers ---
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

# --- Η κύρια επεξεργασία / συνδιασμός των βημάτων ---
def detect_obstacles_withLiDAR(image:          np.ndarray,
                               points:         np.ndarray,
                               P2:             np.ndarray,
                               Tr_velo_to_cam: np.ndarray) -> tuple:
    '''
    Συνδιασμός Βi και ανίχνευσης εμποδίων [Bii] με LiDAR.
    
    Returns:
    - final_image:   Εικόνα που περιέχει τον δρόμο και τα εμπόδια!
    - clusters:      Dict με τα clusters των εμποδίων.
    - ground_points: Το pcd του δρόμου που βρέθηκε βάση της Bi ανίχνευσης.
    '''
    (road_mask, ground_points, ground_plane) = my_road_from_pcd_is(
        points, Tr_velo_to_cam, P2, image.shape
    ) # Δεν γυρνά τα συνολικά σημεία που είναι ορατά από την κάμερα...

    obstacles_raw = filter_points_near_plane(
        filter_visible_points(
            points, Tr_velo_to_cam, P2, image.shape
        ), # ... οπότε, under the hood, εκτελείται 2 φορές...
        ground_plane, 
        max_dist = CONFIG['ground_removal']['non_ground_max_dist'], 
        invert   = True
    )

    obstacles_filtered = filter_by_height(
        obstacles_raw,
        ground_plane,
        min_h = CONFIG['height_filter']['min_height'],
        max_h = CONFIG['height_filter']['max_height']
    )

    labels   = cluster_obstacles_dbscan(obstacles_filtered)
    clusters = group_clusters(obstacles_filtered, labels)

    # --- Ζωγραφικηηή! ---

    # Εμφάνιση του δρόμου στην εικόνα
    overlay = overlay_mask(
        image, road_mask, color = (255, 0, 0), alpha = 0.5
    )

    # Προβολή των clusters στην εικόνα με dots
    colored_dots = project_clusters_with_dots(
        clusters, Tr_velo_to_cam, P2, image.shape
    )
    final_image = visualize_results(overlay, colored_dots)

    return (final_image, clusters, ground_points);

def main():
    base_dir = os.path.dirname(__file__)

    image_type   = 'um'
    dataset_type = 'testing'
    dataset_type = 'training'

    calib_path = os.path.join(base_dir, '..', 'calibration_KITTI.txt')
    (Tr_velo_to_cam, P2) = load_calibration(calib_path)
    
    for i in range(94):
        general_name_file = (f'{image_type}_0000{i}' if i > 9 \
                             else f'{image_type}_00000{i}')

        bin_path = os.path.join(
            base_dir, '..', '..',
            'KITTI', 'data_road_velodyne', dataset_type, 'velodyne',
            f'{general_name_file}.bin'
        )
        img_path = os.path.join(
            base_dir, '..', '..',
            'KITTI', 'data_road', dataset_type, 'image_2',
            f'{general_name_file}.png'
        )
        if not (os.path.isfile(bin_path) and \
                os.path.isfile(img_path)):
            print(f'Πρόβλημα με το {general_name_file}')
            continue;

        image = cv2.imread(img_path)
        points = load_velodyne_bin(bin_path)

        start = time()
        (result_image, _, _) = detect_obstacles_withLiDAR(
            image, points, P2, Tr_velo_to_cam
        )
        print(f'Διάρκεια εκτέλεσης: {time() - start:.2f} sec / {general_name_file}')

        cv2.imshow('DashCam', result_image)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
