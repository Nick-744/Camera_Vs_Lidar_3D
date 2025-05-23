import sys, os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
) # Για την εισαγωγή του Ai_road_finder!
from Ai_road_finder.Ai_from_disparity import *

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from time import time
import open3d as o3d
import numpy as np
import cv2

def parse_kitti_calib(file_path: str) -> dict:
    calib = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue;
            key, value = line.strip().split(':', 1)
            calib[key.strip()] = float(value.strip())

    required_keys = ['f', 'cx', 'cy', 'cx_prime', 'Tx']
    for key in required_keys:
        if key not in calib:
            raise ValueError(
                f'Λείπει η τιμή του {key} από το αρχείο {file_path}!'
            );

    return calib;

def cluster_obstacles_dbscan(points:      np.ndarray,
                             eps:         float = 0.3,
                             min_samples: int = 20,
                             show:        bool = False) -> np.ndarray:
    clustering = DBSCAN(
        eps = eps,
        min_samples = min_samples
    ).fit(points)
    labels = clustering.labels_

    if show:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        colors = plt.get_cmap('tab20')(labels % 20)[:, :3] # Color map
        colors[labels < 0] = [0, 0, 0] # noise = black
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])

    return labels;

def hybrid_cluster(points: np.ndarray,
                   coarse_k: int = 100,
                   db_eps: float = 0.3,
                   db_min_samples: int = 20) -> np.ndarray:
    '''
    Συνδυασμός MiniBatchKMeans και DBSCAN για γρηγορότερο clustering!
    '''
    # MiniBatchKMeans για γρήγορη αρχική ομαδοποίηση
    kmeans = MiniBatchKMeans(
        n_clusters = coarse_k, batch_size = 1000
    ).fit(points)
    labels_kmeans = kmeans.labels_

    # DBSCAN για λεπτομερή ομαδοποίηση!
    final_labels = -np.ones(len(points), dtype = int)
    label_id = 0

    for k in np.unique(labels_kmeans):
        cluster_pts = points[labels_kmeans == k]
        if len(cluster_pts) < db_min_samples:
            continue;

        sub_labels = DBSCAN(
            eps = db_eps,
            min_samples = db_min_samples
        ).fit(cluster_pts).labels_
        mask = sub_labels != -1

        # Ενημέρωση ετικετών για τα σημεία του cluster
        target_indices = np.where(labels_kmeans == k)[0][mask]
        final_labels[target_indices] = label_id + sub_labels[mask]
        label_id += sub_labels.max() + 1

    return final_labels;

def filter_by_height(points: np.ndarray,
                     plane:  tuple,
                     min_h:  float = 0.15,
                     max_h:  float = 2.) -> np.ndarray:
    '''
    Φιλτράρει σημεία με βάση το κατακόρυφο ύψος τους από το επίπεδο.
    '''
    (a, b, c, d) = plane

    normal = np.array([a, b, c], dtype = np.float32)
    norm = np.linalg.norm(normal)
    dists = np.abs((points @ normal + d) / norm)

    mask = (dists >= min_h) & (dists <= max_h)

    return points[mask];

def group_clusters(points: np.ndarray,
                   labels: np.ndarray) -> dict[int, np.ndarray]:
    '''
    Ομαδοποιεί τα σημεία σε clusters με βάση τις ετικέτες τους!
    '''
    valid_mask = labels != -1
    filtered_points = points[valid_mask]
    filtered_labels = labels[valid_mask]

    unique_labels = np.unique(filtered_labels)
    clusters = {label: filtered_points[filtered_labels == label] \
                for label in unique_labels}
    
    return clusters;

def project_to_image(clusters:     dict,
                     calib:        dict,
                     shape:        tuple,
                     min_box_area: int = 30,
                     crop_bottom:  bool = True) -> list:
    '''
    Προβάλλει τα 3D σημεία του κάθε cluster
    στην εικόνα και επιστρέφει bounding boxes.
    '''
    (f, cx, cy) = calib['f'], calib['cx'], calib['cy']
    (h, w) = shape[:2]
    v_offset = h // 2 if crop_bottom else 0

    boxes = []
    for cluster in clusters.values():
        cluster = np.asarray(cluster, dtype = np.float32)
        if cluster.shape[0] < 3:
            continue;

        (x, y, z) = (cluster[:, 0], cluster[:, 1], cluster[:, 2])
        valid = (z > 0.1) & (z < 40)
        if not np.any(valid):
            continue;

        z = z[valid]
        u = (f * x[valid] / z) + cx
        v = (f * y[valid] / z) + cy + v_offset

        finite = np.isfinite(u) & np.isfinite(v)
        (u, v) = (u[finite], v[finite])

        u = np.round(u).astype(int)
        v = np.round(v).astype(int)

        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if np.count_nonzero(in_bounds) < 3:
            continue;

        (u, v) = (u[in_bounds], v[in_bounds])

        (x1, y1) = (u.min(), v.min())
        (x2, y2) = (u.max(), v.max())

        (width, height) = (x2 - x1, y2 - y1)
        area = width * height
        if area >= min_box_area:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return boxes;

def draw_bboxes(img:   np.ndarray,
                boxes: list,
                color: tuple = (0, 255, 0),
                vertical_offset: int = 0) -> None:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(
            img,
            (x1, y1 + vertical_offset),
            (x2, y2 + vertical_offset),
            color,
            2
        )
    
    return;

def filter_boxes_by_road_mask(boxes:              list,
                              road_mask:          np.ndarray,
                              threshold:          float = 0.2,
                              vertical_offset:    int = 0,
                              dilate_kernel_size: int = 11) -> list:
    '''
    Κρατά μόνο τα boxes που επικαλύπτονται με τη
    road_mask πάνω από το threshold ποσοστό!
    '''
    (h, w) = road_mask.shape
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated = cv2.dilate(
        road_mask.astype(np.uint8),
        kernel,
        iterations = 1
    )

    filtered = []
    for (x1, y1, x2, y2) in boxes:
        (y1_shifted, y2_shifted) = (
            y1 + vertical_offset,
            y2 + vertical_offset
        )
        x1 = np.clip(x1, 0, w - 1)
        x2 = np.clip(x2, 0, w - 1)
        y1_shifted = np.clip(y1_shifted, 0, h - 1)
        y2_shifted = np.clip(y2_shifted, 0, h - 1)

        if (x2 <= x1) or (y2_shifted <= y1_shifted):
            continue;

        box_mask = dilated[y1_shifted:y2_shifted, x1:x2]
        area = box_mask.size
        if area == 0:
            continue;

        overlap = np.count_nonzero(box_mask)
        overlap_ratio = overlap / area

        if overlap_ratio >= threshold:
            filtered.append([x1, y1, x2, y2])

    return filtered;

def detect_obstacles(left_color:     np.ndarray,
                     left_gray:      np.ndarray,
                     right_gray:     np.ndarray,
                     original_shape: tuple,
                     calib:          dict,
                     fast:           bool = True) -> tuple:
    '''
    Συνδιασμός όλων των βημάτων για την ανίχνευση εμποδίων!

    Σημείωση: Η σμίκρυνση της ανάλυσης των εικόνων πριν τον
              υπολογισμό του disparity δεν οδηγεί απαραίτητα
              σε εξίσου καλά αποτελέσματα και σε ορισμένες
              περιπτώσεις μπορεί να μην είναι καν πιο γρήγορη!

            **Τελικά, απλά γίνεται επεξεργασία του τμήματος
              ενδιαφέροντος της εικόνας, δηλαδή κάτω μισό!
    '''
    disparity = compute_disparity(left_gray, right_gray)
    pcd = point_cloud_from_disparity(disparity, calib)

    (obstacle_pts, ground_pts, plane) = ransac_ground(pcd)
    filtered_pts = filter_by_height(
        obstacle_pts,
        plane,
        min_h = 0.4,
        max_h = 2.
    ) # ΠΑΡΑ ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ!!!

    labels = (hybrid_cluster(filtered_pts) if fast else \
              cluster_obstacles_dbscan(filtered_pts))
    clusters = group_clusters(filtered_pts, labels)

    boxes = project_to_image(
        clusters,
        calib,
        left_color.shape,
        min_box_area = 300,
    )

    # Φιλτράρισμα boxes με βάση road mask!
    road_mask = project_points_to_mask(
        ground_pts, calib, shape = original_shape
    )
    road_mask_cleaned = post_process_mask(road_mask) # Αναγνώριση δρόμου
    boxes = filter_boxes_by_road_mask(
        boxes,
        road_mask_cleaned,
        threshold = 0.03,
        vertical_offset = left_color.shape[0] // 2,
        dilate_kernel_size = 11
    )

    return (boxes, road_mask, road_mask_cleaned);

def main():
    base_dir = os.path.dirname(__file__)
    dataset_type = 'training'

    calib_path = os.path.join(base_dir, 'calibration_KITTI.txt')
    calib = parse_kitti_calib(calib_path)

    for idx in range(0, 94):
        image_name = (f'um_0000{idx}.png' if idx > 9 \
                      else f'um_00000{idx}.png')

        left_path = os.path.join(
            base_dir,
            '..', '..',
            'KITTI',
            'data_road',
            dataset_type,
            'image_2',
            image_name
        )
        right_path = os.path.join(
            base_dir,
            '..', '..',
            'KITTI',
            'data_road_right',
            dataset_type,
            'image_3',
            image_name
        )

        left_color = cv2.imread(left_path)
        left_gray  = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if (left_color is None) or \
            (left_gray is None) or \
            (right_gray is None):
            print(f'Σφάλμα κατά την φόρτωση των εικόνων {image_name}!')
            continue;
        
        left_color_cropped = crop_bottom_half(left_color)
        left_gray_cropped  = crop_bottom_half(left_gray)
        right_gray_cropped = crop_bottom_half(right_gray)

        start = time()
        (boxes, _, road_mask_cleaned) = detect_obstacles(
            left_color_cropped,
            left_gray_cropped,
            right_gray_cropped,
            original_shape = left_color.shape,
            calib = calib,
            fast = True
        )
        print(f'Χρόνος εκτέλεσης: {time() - start:.2f} sec')

        # Ζωγραφικηηηή
        left_color = overlay_mask(left_color, road_mask_cleaned)

        draw_bboxes(
            left_color,
            boxes,
            vertical_offset = left_color.shape[0] // 4,
        )

        cv2.imshow('Stereo-Only Obstacle Detection', left_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
