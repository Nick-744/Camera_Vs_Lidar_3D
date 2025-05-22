import os
import cv2
import numpy as np
import open3d as o3d
from time import time
from sklearn.cluster import DBSCAN
from Bi_road_detection_pcd import *

def ransac_ground_removal(points: np.ndarray,
                          distance_threshold: float = 0.02,
                          num_iterations: int = 10000) -> tuple:
    (_, ground_plane) = detect_ground_plane(
        points,
        distance_threshold = distance_threshold,
        num_iterations     = num_iterations,
        show = False
    )
    non_ground = filter_points_near_plane(points, ground_plane, max_dist=0.3, invert=True)
    ground     = filter_points_near_plane(points, ground_plane, max_dist=0.15)
    return non_ground, ground, ground_plane

def filter_by_height(points: np.ndarray,
                     plane: tuple,
                     min_h: float = 0.3,
                     max_h: float = 2.5) -> np.ndarray:
    (a, b, c, d) = plane
    norm = np.linalg.norm([a, b, c])
    dists = np.abs(points @ np.array([a, b, c]) + d) / norm
    return points[(dists >= min_h) & (dists <= max_h)]

def cluster_obstacles_dbscan(points: np.ndarray,
                             eps: float = 0.3,
                             min_samples: int = 10) -> np.ndarray:
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_

def group_clusters(points, labels):
    clusters = {}
    for pt, lbl in zip(points, labels):
        if lbl == -1: continue
        clusters.setdefault(lbl, []).append(pt)
    return {k: np.array(v) for k, v in clusters.items()}

def project_to_image(clusters: dict,
                     Tr_velo_to_cam: np.ndarray,
                     P2: np.ndarray,
                     shape: tuple,
                     min_box_area: int = 30) -> list:
    (h, w) = shape[:2]
    boxes = []

    for cluster in clusters.values():
        (_, u, v, _) = project_lidar_to_image(cluster, Tr_velo_to_cam, P2)
        uvs = np.stack([u, v], axis=1)

        # Keep only in-image projections
        uvs = uvs[(uvs[:,0] >= 0) & (uvs[:,0] < w) & (uvs[:,1] >= 0) & (uvs[:,1] < h)]
        if len(uvs) < 3:
            continue

        x1, y1 = np.min(uvs, axis=0)
        x2, y2 = np.max(uvs, axis=0)
        area = (x2 - x1) * (y2 - y1)
        if area >= min_box_area:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return boxes

def draw_bboxes(img: np.ndarray,
                boxes: list,
                color: tuple = (0, 255, 0)) -> None:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def filter_points_near_plane(points: np.ndarray,
                             plane_model: tuple,
                             max_dist: float = 0.15,
                             invert: bool = False) -> np.ndarray:
    '''
    Κρατάει (ή απορρίπτει αν invert=True) σημεία κοντά στο επίπεδο/δρόμο.

    - plane_model:
        Το επίπεδο (a, b, c, d) από RANSAC: ax + by + cz + d = 0.
    - max_dist:
        Μέγιστη απόσταση για να θεωρηθεί στο επίπεδο.
    - invert:
        Αν True, κρατά όσα είναι ΜΑΚΡΙΑ από το επίπεδο.
    '''
    (a, b, c, d) = plane_model
    normal = np.array([a, b, c])
    distances = np.abs(points @ normal + d) / np.linalg.norm(normal)

    return points[distances > max_dist] if invert else points[distances < max_dist]

def main():
    base_dir = os.path.dirname(__file__)
    for i in range(10, 50):
        name = f'um_0000{i}'
        img_path = os.path.join(base_dir, '..', '..', 'KITTI', 'data_road', 'training', 'image_2', f'{name}.png')
        bin_path = os.path.join(base_dir, '..', '..', 'KITTI', 'data_road_velodyne', 'training', 'velodyne', f'{name}.bin')
        calib_path = os.path.join(base_dir, '..', '..', 'KITTI', 'data_road', 'training', 'calib', f'{name}.txt')

        if not (os.path.exists(img_path) and os.path.exists(bin_path) and os.path.exists(calib_path)):
            print(f'Λείπει αρχείο: {name}')
            continue

        image = cv2.imread(img_path)
        points = load_velodyne_bin(bin_path)
        Tr_velo_to_cam, P2 = load_calibration(calib_path)

        # Keep only visible points
        visible = filter_visible_points(points, Tr_velo_to_cam, P2, image.shape)

        # Ground removal
        obstacles_raw, ground, plane = ransac_ground_removal(visible)

        # Height filter
        obstacles_filtered = filter_by_height(obstacles_raw, plane)

        # DBSCAN clustering
        labels = cluster_obstacles_dbscan(obstacles_filtered)
        clusters = group_clusters(obstacles_filtered, labels)

        # Project clusters to 2D image
        boxes = project_to_image(clusters, Tr_velo_to_cam, P2, image.shape)

        # Optional: filter boxes with road mask
        # road_mask = project_points_to_image(ground, Tr_velo_to_cam, P2, image.shape)
        # boxes = filter_boxes_by_road_mask(boxes, road_mask, threshold=0.04)

        # Draw boxes
        draw_bboxes(image, boxes)

        cv2.imshow("LiDAR Obstacle Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
