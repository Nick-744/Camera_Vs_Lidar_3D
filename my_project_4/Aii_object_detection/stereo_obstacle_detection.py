import os
import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def parse_kitti_calib(file_path: str) -> dict:
    calib = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            calib[key.strip()] = float(value.strip())

    required_keys = ['f', 'cx', 'cy', 'cx_prime', 'Tx']
    for key in required_keys:
        if key not in calib:
            raise ValueError(
                f"Λείπει η τιμή του {key} από το αρχείο {file_path}"
            );

    return calib;

def ransac_ground_removal(
    points: np.ndarray,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 2000,
    show: bool = True
) -> tuple:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Εφαρμογή RANSAC για να βρούμε επίπεδο (έδαφος)
    (plane_model, inliers) = pcd.segment_plane(
        distance_threshold = distance_threshold,
        ransac_n           = ransac_n,
        num_iterations     = num_iterations
    )

    ground_points = pcd.select_by_index(inliers)
    obstacle_points = pcd.select_by_index(inliers, invert = True)

    if show:
        ground_points.paint_uniform_color([0.0, 1.0, 0.0])   # Πράσινο
        obstacle_points.paint_uniform_color([1.0, 0.0, 0.0]) # Κόκκινο

        o3d.visualization.draw_geometries([ground_points, obstacle_points])

    return (
        np.asarray(obstacle_points.points),
        np.asarray(ground_points.points),
        plane_model
    );

def cluster_obstacles_dbscan(
    points: np.ndarray,
    eps: float = 0.2,
    min_samples: int = 10,
    show: bool = True
) -> np.ndarray:
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if show:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        colors = plt.get_cmap("tab20")(labels % 20)[:, :3]  # use color map
        colors[labels < 0] = [0, 0, 0]  # noise = black
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])

    return labels;

def point_cloud_from_disparity(
    disparity,
    calib,
    left_image_color,
    show = False
):
    f = calib['f']
    cx = calib['cx']
    cy = calib['cy']
    Tx = calib['Tx']

    h, w = disparity.shape
    mask = disparity > 0

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    us = us[mask]
    vs = vs[mask]
    ds = disparity[mask]

    Z = f * Tx / ds
    X = (us - cx) * Z / f
    Y = (vs - cy) * Z / f
    points = np.stack((X, Y, Z), axis=-1)

    colors = left_image_color[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

    if show:
        o3d.visualization.draw_geometries([pcd])
    
    return pcd;

def filter_by_height(points, plane, min_h=0.15, max_h=2.0):
    a, b, c, d = plane
    normal = np.linalg.norm([a, b, c])
    filtered = []
    for pt in points:
        height = abs((a * pt[0] + b * pt[1] + c * pt[2] + d) / normal)
        if min_h <= height <= max_h:
            filtered.append(pt)

    return np.array(filtered);

def group_clusters(points, labels):
    clusters = {}
    for pt, label in zip(points, labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(pt)

    return {k: np.array(v) for k, v in clusters.items()};

def project_to_image(clusters, calib, shape, min_box_area=30, debug_img=None):
    f, cx, cy = calib['f'], calib['cx'], calib['cy']
    h, w = shape[:2]
    boxes = []

    projected_clusters = 0

    for cluster_id, cluster in clusters.items():
        uvs = []

        for x, y, z in cluster:
            if not (0.1 < z < 40):
                continue
            u = (f * x / z) + cx
            v = (f * y / z) + cy
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            u, v = int(round(u)), int(round(v))
            if 0 <= u < w and 0 <= v < h:
                uvs.append((u, v))
                if debug_img is not None:
                    cv2.circle(debug_img, (u, v), 1, (0, 0, 255), -1)

        if len(uvs) < 3:
            continue

        uvs = np.array(uvs)
        x1, y1 = uvs.min(axis=0)
        x2, y2 = uvs.max(axis=0)
        width, height = x2 - x1, y2 - y1
        area = width * height

        if area >= min_box_area:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            projected_clusters += 1

    return boxes;

def draw_bboxes(img, boxes, color=(0, 255, 0), label = ''):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return;

def compute_disparity(left_gray, right_gray):
    window_size = 9  # Smaller is more accurate for thin structures
    num_disp = 16 * 12  # Must be divisible by 16

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size * window_size,
        P2=32 * 3 * window_size * window_size,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Optional: Clip extreme disparities (can help visualization and stability)
    disparity[disparity < 0] = 0

    return disparity;

def main():
    base_dir = os.path.dirname(__file__)

    for idx in range(14, 25):
        image_name = f'um_0000{idx}.png'

        left_path = os.path.join(
            base_dir,
            '..',
            'KITTI',
            'data_road',
            'testing',
            'image_2',
            image_name
        )
        right_path = os.path.join(
            base_dir,
            '..',
            'KITTI',
            'data_road_right',
            'testing',
            'image_3',
            image_name
        )
        calib_path = os.path.join(base_dir, 'calibration_KITTI.txt')

        left_color = cv2.imread(left_path)
        left_gray = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        calib = parse_kitti_calib(calib_path)

        disparity = compute_disparity(left_gray, right_gray)
        pcd = point_cloud_from_disparity(
            disparity,
            calib,
            left_color,
            show = False
        )
        raw_points = np.asarray(pcd.points)

        obstacle_pts, _, plane = ransac_ground_removal(
            raw_points,
            distance_threshold = 0.02,
            ransac_n = 3,
            num_iterations = 5000,
            show = False
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
            min_samples = 30,
            show = False
        )
        clusters = group_clusters(filtered_pts, labels)

        debug_img = left_color.copy()
        boxes = project_to_image(
            clusters,
            calib,
            left_color.shape,
            min_box_area = 100,
            debug_img = debug_img
        )

        vis = left_color.copy()
        draw_bboxes(vis, boxes)

        cv2.imshow("Stereo-Only Obstacle Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
