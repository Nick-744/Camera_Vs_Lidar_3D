import os
import cv2
import numpy as np
import open3d as o3d
from time import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

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
                f"Λείπει η τιμή του {key} από το αρχείο {file_path}!"
            );

    return calib;

def ransac_ground_removal(points:             np.ndarray,
                          distance_threshold: float = 0.01,
                          ransac_n:           int = 3,
                          num_iterations:     int = 2000,
                          show:               bool = False) -> tuple:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Εφαρμογή RANSAC για να βρούμε τον δρόμο/επίπεδο
    (plane_model, inliers) = pcd.segment_plane(
        distance_threshold = distance_threshold,
        ransac_n           = ransac_n,
        num_iterations     = num_iterations
    )

    ground_points = pcd.select_by_index(inliers)
    obstacle_points = pcd.select_by_index(inliers, invert = True)
    if show:
        ground_points.paint_uniform_color(  [0., 1., 0.]) # Πράσινο
        obstacle_points.paint_uniform_color([1., 0., 0.]) # Κόκκινο
        o3d.visualization.draw_geometries([ground_points, obstacle_points])

    return (
        np.asarray(obstacle_points.points),
        np.asarray(ground_points.points),
        plane_model
    );

def cluster_obstacles_dbscan(points:      np.ndarray,
                             eps:         float = 0.2,
                             min_samples: int = 10,
                             show:        bool = False) -> np.ndarray:
    clustering = DBSCAN(
        eps = eps,
        min_samples = min_samples
    ).fit(points)
    labels = clustering.labels_

    if show:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        colors = plt.get_cmap("tab20")(labels % 20)[:, :3] # Color map
        colors[labels < 0] = [0, 0, 0] # noise = black
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])

    return labels;

def point_cloud_from_disparity(
    disparity:        np.ndarray,
    calib:            dict,
    left_image_color: np.ndarray,
    show:             bool = False
) -> o3d.geometry.PointCloud:
    (f, cx, cy, Tx) = (
        calib['f'],
        calib['cx'],
        calib['cy'],
        calib['Tx']
    )

    (h, w) = disparity.shape
    mask = disparity > 0

    (us, vs) = np.meshgrid(np.arange(w), np.arange(h))
    (us, vs) = (us[mask], vs[mask])
    ds = disparity[mask]

    Z = f * Tx / ds
    X = (us - cx) * Z / f
    Y = (vs - cy) * Z / f
    points = np.stack((X, Y, Z), axis = -1)

    colors = left_image_color[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.)
    if show:
        o3d.visualization.draw_geometries([pcd])
    
    return pcd;

def filter_by_height(points: np.ndarray,
                     plane:  tuple,
                     min_h:  float = 0.15,
                     max_h:  float = 2.) -> np.ndarray:
    (a, b, c, d) = plane
    normal = np.linalg.norm([a, b, c])
    filtered = []
    for pt in points:
        height = abs((a * pt[0] + b * pt[1] + c * pt[2] + d) / normal)
        if min_h <= height <= max_h:
            filtered.append(pt)

    return np.array(filtered);

def group_clusters(points: np.ndarray, labels: np.ndarray) -> dict:
    clusters = {}
    for (pt, label) in zip(points, labels):
        if label == -1:
            continue;
        clusters.setdefault(label, []).append(pt)

    return {k: np.array(v) for k, v in clusters.items()};

def project_to_image(clusters:     dict,
                     calib:        dict,
                     shape:        tuple,
                     min_box_area: int = 30) -> list:
    (f, cx, cy) = calib['f'], calib['cx'], calib['cy']
    (h, w) = shape[:2]
    boxes = []

    for (_, cluster) in clusters.items():
        uvs = []
        for (x, y, z) in cluster:
            if not (0.1 < z < 40):
                continue;
            (u, v) = ((f * x / z) + cx, (f * y / z) + cy)
            if not np.isfinite(u) or not np.isfinite(v):
                continue;
            (u, v) = (int(round(u)), int(round(v)))
            if (0 <= u < w) and (0 <= v < h):
                uvs.append((u, v))

        if len(uvs) < 3:
            continue;

        uvs = np.array(uvs)
        (x1, y1) = uvs.min(axis=0)
        (x2, y2) = uvs.max(axis=0)
        (width, height) = (x2 - x1, y2 - y1)
        area = width * height

        if area >= min_box_area:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return boxes;

def draw_bboxes(img:   np.ndarray,
                boxes: list,
                color: tuple = (0, 255, 0)) -> None:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    return;

def compute_disparity(left_gray:  np.ndarray,
                      right_gray: np.ndarray) -> np.ndarray:
    window_size = 9
    num_disp    = 16 * 6 # Πρέπει να είναι πολλαπλάσιο του 16!

    stereo = cv2.StereoSGBM_create(
        minDisparity      = 0,
        numDisparities    = num_disp,
        blockSize         = window_size,
        P1                = 8 * 3 * window_size * window_size,
        P2                = 32 * 3 * window_size * window_size,
        disp12MaxDiff     = 1,
        uniquenessRatio   = 15,
        speckleWindowSize = 100,
        speckleRange      = 32,
        preFilterCap      = 63,
        mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(
        left_gray,
        right_gray
    ).astype(np.float32) / 16.
    
    # Εξασφαλίζουμε ότι το disparity είναι θετικό
    disparity[disparity < 0] = 0

    return disparity;

def filter_boxes_by_road_mask(boxes:              list,
                              road_mask:          np.ndarray,
                              threshold:          float = 0.2,
                              dilate_kernel_size: int = 11) -> list:
    '''
    Κρατά μόνο τα boxes που επικαλύπτονται με τη
    road_mask πάνω από το threshold ποσοστό!
    '''
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(
        road_mask.astype(np.uint8),
        kernel,
        iterations = 1
    )

    filtered_boxes = []
    for box in boxes:
        (x1, y1, x2, y2) = box
        (x1, y1) = max(x1, 0), max(y1, 0)
        (x2, y2) = min(x2, road_mask.shape[1]), min(y2, road_mask.shape[0])

        box_area = (x2 - x1) * (y2 - y1)
        if box_area == 0:
            continue;

        box_mask = dilated_mask[y1:y2, x1:x2]
        overlap = np.count_nonzero(box_mask)
        overlap_ratio = overlap / box_area

        if overlap_ratio >= threshold:
            filtered_boxes.append(box)

    return filtered_boxes;

def ground_mask_from_points(ground_pts:  np.ndarray,
                            calib:       dict,
                            image_shape: tuple,
                            blur_kernel: int = 15) -> np.ndarray:
    (f, cx, cy) = (calib['f'], calib['cx'], calib['cy'])
    (h, w) = image_shape[:2]
    mask = np.zeros((h, w), dtype = np.uint8)

    for (x, y, z) in ground_pts:
        if not (0.1 < z < 40):
            continue;
        u = int(round((f * x / z) + cx))
        v = int(round((f * y / z) + cy))
        if (0 <= u < w) and (0 <= v < h):
            mask[v, u] = 1

    # Κυρίως λόγω θορύβου
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 2)

    # Κρατάμε μόνο το μεγαλύτερο συνεκτικό τμήμα της μάσκας!
    (
        num_labels,
        labels,
        stats,
        _
    ) = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    if num_labels <= 1: # Για να μην γίνει πατάτα...
        return mask;

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    binary = (labels == largest).astype(np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.GaussianBlur(
        binary.astype(np.float32), (blur_kernel, blur_kernel), 0
    )
    
    return (binary > 0.1).astype(np.uint8);

def is_obstacle_on_road(cluster_points: np.ndarray,
                        ground_points:  np.ndarray,
                        plane_model:    tuple,
                        percentage:     float = 1.,
                        xy_margin:      float = 0.) -> bool:
    '''
    Ελέγχει αν το cluster είναι πάνω στον δρόμο (με 3D επεξεργασία):
    - Προβάλλει τα σημεία του cluster πάνω στο επίπεδο του δρόμου.
    - Υπολογίζει αν βρίσκονται εντός των ορίων XZ των ground
      points, με περιθώριο.
    '''
    (a, b, c, d) = plane_model
    n = np.array([a, b, c])
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        return False;
    
    n = n / n_norm

    # Όρια δρόμου στο X-Z επίπεδο
    ground_xz = ground_points[:, [0, 2]]
    (min_x, max_x) = (ground_xz[:, 0].min(), ground_xz[:, 0].max())
    (min_z, max_z) = (ground_xz[:, 1].min(), ground_xz[:, 1].max())

    projected = []
    for p in cluster_points:
        # Υπολογισμός προβολής
        dist_to_plane = np.dot(n, p) + d
        p_proj = p - dist_to_plane * n
        projected.append(p_proj[[0, 2]]) # Κρατάμε ΜΟΝΟ X, Z
    projected = np.array(projected)

    # Αριθμός σημείων (cluster) που είναι εντός του δρόμου/ορίου
    in_bounds = np.logical_and.reduce((
        projected[:, 0] >= (min_x - xy_margin),
        projected[:, 0] <= (max_x + xy_margin),
        projected[:, 1] >= (min_z - xy_margin),
        projected[:, 1] <= (max_z + xy_margin)
    ))

    ratio = np.count_nonzero(in_bounds) / len(projected)

    return ratio >= percentage;

def detect_obstacles(left_color:       np.ndarray,
                     left_gray:        np.ndarray,
                     right_gray:       np.ndarray,
                     calib:            dict,
                     road_filter_mode: str = '2d_mask',
                     debug:            bool = False) -> tuple:
    '''
    Συνδιασμός όλων των βημάτων για την ανίχνευση εμποδίων!

    - road_filter_mode: '2d_mask' ή '3d_plane'. Γενικά, το 2d_mask
                        δουλεύει καλύτερα/γρηγορότερα.

    Σημείωση: Η σμίκρυνση της ανάλυσης των εικόνων πριν τον
              υπολογισμό του disparity δεν οδηγεί απαραίτητα
              σε εξίσου καλά αποτελέσματα και σε ορισμένες
              περιπτώσεις μπορεί να μην είναι καν πιο γρήγορη!
    '''
    disparity = compute_disparity(left_gray, right_gray)
    pcd = point_cloud_from_disparity(
        disparity,
        calib,
        left_color,
        show = debug
    )
    raw_points = np.asarray(pcd.points)

    (obstacle_pts, ground_pts, plane) = ransac_ground_removal(
        raw_points,
        distance_threshold = 0.02,
        ransac_n = 3,
        num_iterations = 2000,
        show = debug
    )
    filtered_pts = filter_by_height(
        obstacle_pts,
        plane,
        min_h = 0.4,
        max_h = 2.
    )

    labels = cluster_obstacles_dbscan(
        filtered_pts,
        eps = 0.3,
        min_samples = 20,
        show = debug
    )
    clusters = group_clusters(filtered_pts, labels)

    if road_filter_mode == '3d_plane':
        clusters = {
            i: cluster for (i, cluster) in clusters.items()
            if is_obstacle_on_road(cluster, ground_pts, plane)
        }

    boxes = project_to_image(
        clusters,
        calib,
        left_color.shape,
        min_box_area = 200
    )

    # Φιλτράρισμα boxes με βάση road mask!
    road_mask = None
    if road_filter_mode == '2d_mask':
        road_mask = ground_mask_from_points(
            ground_pts,
            calib,
            left_color.shape
        )
        boxes = filter_boxes_by_road_mask(
            boxes,
            road_mask,
            threshold = 0.04,
            dilate_kernel_size = 11
        )

    return (boxes, road_mask);

def main():
    base_dir = os.path.dirname(__file__)
    dataset_type = 'training'

    for idx in range(10, 50):
        image_name = f'um_0000{idx}.png'

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
        calib_path = os.path.join(base_dir, 'calibration_KITTI.txt')
        calib = parse_kitti_calib(calib_path)

        left_color = cv2.imread(left_path)
        left_gray  = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if (left_color is None) or \
            (left_gray is None) or \
            (right_gray is None):
            print(f'Σφάλμα κατά την φόρτωση των εικόνων {image_name}!')
            continue;

        start = time()
        (boxes, road_mask) = detect_obstacles(
            left_color,
            left_gray,
            right_gray,
            calib
        )
        print(f'Χρόνος εκτέλεσης: {time() - start:.2f} sec')

        # Ζωγραφικηηηή
        draw_bboxes(left_color, boxes)

        cv2.imshow("Stereo-Only Obstacle Detection", left_color)
        # cv2.imshow("Road Mask", road_mask * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return;

if __name__ == "__main__":
    main()
