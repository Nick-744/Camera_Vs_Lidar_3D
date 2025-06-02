import sys, os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
) # Για την εισαγωγή του Ai_road_finder!
from Ai_road_finder.Ai_from_disparity import *

from sklearn.cluster import (MiniBatchKMeans, DBSCAN)
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from typing import Dict # For Python 3.7.1 error fix!
from time import time
import open3d as o3d
import numpy as np
import cv2

os.environ['LOKY_MAX_CPU_COUNT'] = '4' # Χαζομάρες για logical cores...

# --- Ανίχνευση εμποδίων ---
def cluster_obstacles_dbscan(points:      np.ndarray,
                             eps:         float = 0.3,
                             min_samples: int = 20,
                             show:        bool = False) -> np.ndarray:
    clustering = DBSCAN(
        eps = eps, min_samples = min_samples
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

def hybrid_cluster_parallel(points:         np.ndarray,
                            coarse_k:       int = 100,
                            db_eps:         float = 0.3,
                            db_min_samples: int = 20,
                            n_jobs:         int = -1) -> np.ndarray:
    '''
    O(n/2) + O(n/2) faster than O(n) λογική!
    Παρόλο που παραμένει O(n) θεωρητικά...
    '''
    kmeans = MiniBatchKMeans(
        n_clusters = coarse_k, batch_size = 3072
    ).fit(points)
    labels_kmeans = kmeans.labels_

    final_labels = -np.ones(len(points), dtype = int)

    # Parallel DBSCAN σε κάθε cluster!
    results = Parallel(n_jobs = n_jobs)(
        delayed(process_cluster)(
            k,
            labels_kmeans,
            points,
            db_eps,
            db_min_samples,
            label_offset = i * 1000
        )
        for (i, k) in enumerate(np.unique(labels_kmeans))
    )

    for (sub_labels, indices) in results:
        final_labels[indices] = np.maximum(
            final_labels[indices], sub_labels
        )

    return final_labels;

def group_clusters(points: np.ndarray,
                   labels: np.ndarray) -> Dict[int, np.ndarray]:
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

def filter_boxes_by_road_mask(boxes:              list,
                              road_mask:          np.ndarray,
                              dilate_kernel_size: int = 10,
                              threshold:          float = 0.2,
                              min_side_contact:   int = 10) -> list:
    '''
    Κρατά μόνο τα boxes που επικαλύπτονται αρκετά με τον δρόμο
    ΚΑΙ έχουν τουλάχιστον 2 πλευρές σε επαφή με τη road_mask.
    Παράλληλα, φιλτράρει και τα boxes που είναι πολύ δυσανάλογα!
    '''
    (h, w) = road_mask.shape
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated = cv2.dilate(
        road_mask.astype(np.uint8), kernel, iterations = 2
    )

    filtered = []
    for (x1, y1, x2, y2) in boxes:
        (x1, x2) = (np.clip(x1, 0, w - 1), np.clip(x2, 0, w - 1))
        (y1, y2) = (np.clip(y1, 0, h - 1), np.clip(y2, 0, h - 1))

        if (x2 <= x1) or (y2 <= y1):
            continue;

        # Προστέθηκε και φίλτρο για δυσανάλογα boxes!
        # Έγινε σε αυτό το σημείο από θέμα ευκολίας για εμένα...
        if (x2 - x1) / (y2 - y1) > 2.5:
            continue;

        box_mask = dilated[y1:y2, x1:x2]
        area = box_mask.size
        if area == 0:
            continue;

        overlap = np.count_nonzero(box_mask)
        overlap_ratio = overlap / area

        sides_touch = 0 # Έλεγξε αν κάποια πλευρά αγγίζει αρκετά!
        if np.count_nonzero(dilated[y1,     x1:x2]) >= min_side_contact:
            sides_touch += 1
        if np.count_nonzero(dilated[y2 - 1, x1:x2]) >= min_side_contact:
            sides_touch += 1
        if np.count_nonzero(dilated[y1:y2,     x1]) >= min_side_contact:
            sides_touch += 1
        if np.count_nonzero(dilated[y1:y2, x2 - 1]) >= min_side_contact:
            sides_touch += 1

        if (overlap_ratio >= threshold) and (sides_touch >= 2):
            filtered.append([x1, y1, x2, y2])

    return filtered;

'''
Έγιναν δοκιμές να εφαρμοστεί DBSCAN (de-noise) αμέσως μετά το RANSAC
με αποτέλεσμα να πάρω μία πολύ καλή αναπαράσταση του χώρου
του δρόμου μόνο με όσα εμπόδια βρίσκονταν πάνω του. Όμως,
λόγω του DBSCAN, η διαδικασία αργούσε πολύ (~ 2 sec)...
Έτσι, κατέληξα ξανά στον συμβιβασμό με τον οποίο ξεκίνησα,
δηλαδή να φιλτράρω τα σημεία με βάση το ύψος τους! Ίσως η
τρισδιάστατη αναπαράσταση που μόλις ανέφερα είχε καλύτερα
αποτελέσματα στον εντοπισμό των εμποδίων, αλλά δεν άξιζε
απλά και μόνο λόγω του χρόνου εκτέλεσης!
'''
def detect_obstacles(left_gray:      np.ndarray,
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

    # Αναγνώριση δρόμου
    (obstacle_pts, ground_pts, plane) = ransac_ground(pcd)

    # ΠΑΡΑ ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ!!! Επιταχύνει την διαδικασία!
    filtered_pts = filter_by_height(
        obstacle_pts, plane, min_h = 0.2, max_h = 2.
    )

    labels = (hybrid_cluster_parallel(filtered_pts) if fast else \
              cluster_obstacles_dbscan(filtered_pts))
    clusters = group_clusters(filtered_pts, labels)

    boxes = project_to_image(
        clusters, calib, original_shape, min_box_area = 300,
    )

    # Φιλτράρισμα boxes με βάση road mask!
    road_mask = project_points_to_mask(ground_pts, calib, original_shape)
    road_mask_cleaned = post_process_mask(road_mask, min_area = 16000)
    boxes = filter_boxes_by_road_mask(
        boxes,
        road_mask_cleaned,
        dilate_kernel_size = 11,
        threshold          = 0.05,
        min_side_contact   = 14 # γύρω στο 14 <-> 16 καλύτερα αποτελέσματα
    )

    return (boxes, road_mask, road_mask_cleaned);

# --- Ανίχνευση εμποδίων με YOLO ---
class YOLODetector:
    ''' Κυρίως για λόγους ταχύτητας κατά την εκτέλεση CARLA! '''
    def __init__(self, model_name: str = 'yolov5s', conf: float = 0.25):
        import torch
        import warnings
        warnings.filterwarnings(
            'ignore',
            category = FutureWarning,
            message  = '.*torch.cuda.amp.autocast.*'
        )

        self.model = torch.hub.load(
            'ultralytics/yolov5', model_name, pretrained = True
        )
        self.model.conf = conf

        # Φιλτράρισμα για τα classes που μας ενδιαφέρουν!
        self.allowed_classes = {
            'person', 'car', 'truck', 'bus', 'motorcycle'
        }

        return;

    def detect(self,
               left_color: np.ndarray,
               left_gray:  np.ndarray,
               right_gray: np.ndarray,
               calib:      dict,) -> tuple:
        '''
        Ανίχνευση εμποδίων μέσω YOLO! Εύρεση μάσκας μέσω disparity.

        Returns:
        - boxes:
        Λίστα με [x1, y1, x2, y2] για κάθε αντικείμενο.
        - road_mask:
        Βάση της υλοποίησης στο Ai_from_disparity.py!
        - road_mask_cleaned:
        Εφαρμογή της συνάρτησης post_process_mask στο road_mask.
        '''
        # Εκτέλεση YOLO
        results = self.model(left_color)
        preds = results.pandas().xyxy[0] # DataFrame

        boxes = []
        for (_, row) in preds.iterrows():
            if row['name'] in self.allowed_classes:
                (x1, y1, x2, y2) = map(int,
                    [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                )
                boxes.append([x1, y1, x2, y2])

        road_mask = detect_ground_mask(
            left_gray,
            right_gray,
            left_color.shape,
            calib,
            crop_bottom = True
        )
        road_mask_cleaned = post_process_mask(
            road_mask, min_area = 15000, kernel_size = 7
        )

        return (boxes, road_mask, road_mask_cleaned);

# --- Helpers ---
def draw_bboxes(img:   np.ndarray,
                boxes: list,
                color: tuple = (0, 255, 0)) -> None:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(
            img, (x1, y1), (x2, y2), color, 2
        )
    
    return;

def process_cluster(k:              int,
                    labels_kmeans:  np.ndarray,
                    points:         np.ndarray,
                    db_eps:         float,
                    db_min_samples: int,
                    label_offset:   int) -> tuple:
    '''
    Βοηθητική συνάρτηση για τον παραλλησμό των clusters!
    '''
    k_mask = (labels_kmeans == k)
    cluster_pts = points[k_mask]

    if len(cluster_pts) < db_min_samples:
        return (np.full(len(cluster_pts), -1), np.flatnonzero(k_mask));

    sub_labels = DBSCAN(
        eps = db_eps, min_samples = db_min_samples
    ).fit(cluster_pts).labels_

    valid_mask = sub_labels != -1
    sub_labels[~valid_mask]  = -1 # Θόρυβος = -1
    sub_labels[valid_mask]  += label_offset

    return (sub_labels, np.flatnonzero(k_mask));

def main():
    base_dir = os.path.dirname(__file__)

    # True για YOLO, False για pure CV approach!
    use_yolo = False
    use_yolo = True
    if use_yolo:
        yolo_detector = YOLODetector(
            model_name = 'yolov5s', conf = 0.25
        )

    image_type = 'um'
    dataset_type = 'testing'
    dataset_type = 'training'

    calib_path = os.path.join(base_dir, '..', 'calibration_KITTI.txt')
    calib      = parse_kitti_calib(calib_path)

    for i in range(94):
        image_name =  (f'{image_type}_0000{i}.png' if i > 9 \
                       else f'{image_type}_00000{i}.png')

        left_path = os.path.join(
            base_dir, '..', '..',
            'KITTI', 'data_road', dataset_type, 'image_2',
            image_name
        )
        right_path = os.path.join(
            base_dir, '..', '..',
            'KITTI', 'data_road_right', dataset_type, 'image_3',
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
        
        left_gray_cropped  = crop_bottom_half(left_gray)
        right_gray_cropped = crop_bottom_half(right_gray)

        start = time()
        if use_yolo:
            (boxes, _, road_mask_cleaned) = yolo_detector.detect(
                left_color,
                left_gray_cropped,
                right_gray_cropped,
                calib
            )
        else:
            (boxes, _, road_mask_cleaned) = detect_obstacles(
                left_gray_cropped,
                right_gray_cropped,
                original_shape = left_color.shape,
                calib          = calib,
                fast           = True
            )
        print(f'Χρόνος εκτέλεσης: {time() - start:.2f} sec')

        # --- Ζωγραφικηηηή ---
        
        # Δρόμος
        left_color = overlay_mask(left_color, road_mask_cleaned)
        
        # Εμπόδια
        draw_bboxes(left_color, boxes)

        cv2.imshow('Stereo-Only Obstacle Detection', left_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
