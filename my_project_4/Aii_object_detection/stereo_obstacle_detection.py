import os
import cv2
import numpy as np
import open3d as o3d

from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.util import img_as_float
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def parse_kitti_calib(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    P2 = None
    P3 = None
    for line in lines:
        if line.startswith("P2:"):
            P2 = np.array(list(map(float, line.strip().split()[1:]))).reshape(3, 4)
        elif line.startswith("P3:"):
            P3 = np.array(list(map(float, line.strip().split()[1:]))).reshape(3, 4)

    if P2 is None or P3 is None:
        raise ValueError('Πρόβλημα στην ανάγνωση των P2 ή P3!');

    f = P2[0, 0]
    cx = P2[0, 2]
    cy = P2[1, 2]
    cx_prime = P3[0, 2]
    Tx = -(P3[0, 3] - P2[0, 3]) / f

    return {
        "f": f,
        "cx": cx,
        "cy": cy,
        "cx_prime": cx_prime,
        "Tx": Tx
    };

def point_cloud_from_disparity(
    disparity: np.ndarray,
    calib: dict,
    left_image_color: np.ndarray,
    show: bool = True
) -> o3d.geometry.PointCloud:
    Q = np.array([
        [1, 0,                0, -calib["cx"]],
        [0, 1,                0, -calib["cy"]],
        [0, 0,                0, + calib["f"]],
        [0, 0, -1 / calib["Tx"],            0]
    ], dtype = np.float32)

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    left_color = cv2.cvtColor(left_image_color, cv2.COLOR_BGR2RGB)

    # Μάσκα έγκυρων disparity τιμών (> 0)
    mask = disparity > 0
    out_points = points_3D[mask]
    out_colors = left_color[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_points)
    pcd.colors = o3d.utility.Vector3dVector(out_colors.astype(np.float32) / 255.0)

    if show:
        o3d.visualization.draw_geometries(
            [pcd],
            zoom  = 0.5,
            front = [0.0, 0.0, -1.0],
            lookat= [0.0, 0.0, 0.0],
            up    = [0.0, -1.0, 0.0]
        )

    return pcd;

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
    print(f"[✓] DBSCAN found {num_clusters} clusters")

    if show:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        colors = plt.get_cmap("tab20")(labels % 20)[:, :3]  # use color map
        colors[labels < 0] = [0, 0, 0]  # noise = black
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])

    return labels;

def average_disparity_by_slic(
    disparity: np.ndarray,
    rgb_image: np.ndarray,
    num_segments: int = 2000,
    compactness: float = 10.,
) -> np.ndarray:
    rgb_float = img_as_float(rgb_image)
    segments = slic(rgb2lab(rgb_float), n_segments=num_segments, compactness=compactness)

    disparity_smoothed = np.zeros_like(disparity)
    for seg_val in np.unique(segments):
        mask = segments == seg_val
        valid_disp = disparity[mask][disparity[mask] > 0]
        if valid_disp.size > 0:
            mean_disp = np.mean(valid_disp)
        else:
            mean_disp = 0.
        disparity_smoothed[mask] = mean_disp

    return disparity_smoothed;

def test():
    base_dir = os.path.dirname(__file__)

    for i in range(19, 20):
        image_name = f'um_0000{i}.png'
        # Φόρτωση του ζεύγους εικόνων
        temp = os.path.join(
            base_dir,
            '..',
            'data_road',
            'training',
            'image_2',
            image_name
        )
        img_file = os.path.abspath(temp)
        left_image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        left_image_color = cv2.imread(img_file)

        temp = os.path.join(
            base_dir,
            '..',
            'data_road_right',
            'training',
            'image_3',
            image_name
        )
        img_file = os.path.abspath(temp)
        right_image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

        # Φόρτωση του calibration file
        temp = os.path.join(
            base_dir,
            '..',
            'data_road',
            'training',
            'calib',
            f'um_0000{i}.txt'
        )
        calib_file = os.path.abspath(temp)
        calib = parse_kitti_calib(calib_file)

        block_size = 10
        stereo = cv2.StereoSGBM_create(
            minDisparity      = 0,
            numDisparities    = 16 * 6,
            blockSize         = block_size,
            # Smoothness Penalty
            P1                = 8 * 3 * block_size*block_size,
            P2                = 32 * 3 * block_size*block_size,
            # Post-Processing
            disp12MaxDiff     = 1,
            uniquenessRatio   = 20, # Μεγαλύτερη τιμή = λιγότερος θόρυβος
            speckleWindowSize = 100,
            speckleRange      = 32,
            preFilterCap      = 63,
            mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        ''' Η συνάρτηση stereo.compute() επιστρέφει τις τιμές disparity 
        σε μορφή fixed-point (Q4), δηλαδή πολλαπλασιασμένες επί 16.
        Για να ανακτήσουμε τις πραγματικές διαφορές θέσης (σε pixels),
        διαιρούμε με το 16.0 ώστε να έχουμε subpixel ακρίβεια! '''
        disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.
        points = point_cloud_from_disparity(
            disparity,
            calib,
            left_image_color,
            show = False
        )

        (obstacle_pts, _, _) = ransac_ground_removal(
            points.points,
            distance_threshold = 0.02,
            ransac_n           = 3,
            num_iterations     = 2000,
            show               = False
        )

        cluster_obstacles_dbscan(obstacle_pts, eps = 0.2, min_samples = 100, show = True)

    return;

def compute_disparity(left_gray, right_gray):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,
        blockSize=9,
        P1=8 * 3 * 9 ** 2,
        P2=32 * 3 * 9 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

def point_cloud_from_disparity(disparity, calib, left_image_color, show=False):
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
    return pcd

def filter_by_height(points, plane, min_h=0.15, max_h=2.0):
    a, b, c, d = plane
    normal = np.linalg.norm([a, b, c])
    filtered = []
    for pt in points:
        height = abs((a * pt[0] + b * pt[1] + c * pt[2] + d) / normal)
        if min_h <= height <= max_h:
            filtered.append(pt)
    return np.array(filtered)

def group_clusters(points, labels):
    clusters = {}
    for pt, label in zip(points, labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(pt)
    return {k: np.array(v) for k, v in clusters.items()}

def project_to_image(clusters, calib, shape, min_box_area=30, debug_img=None):
    f, cx, cy = calib['f'], calib['cx'], calib['cy']
    h, w = shape[:2]
    boxes = []
    total_clusters = len(clusters)

    print(f"[DEBUG] Calibration: f={f:.4f}, cx={cx:.4f}, cy={cy:.4f}")
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
            print(f"[SKIP] Cluster {cluster_id}: only {len(uvs)} valid projections")
            continue

        uvs = np.array(uvs)
        x1, y1 = uvs.min(axis=0)
        x2, y2 = uvs.max(axis=0)
        width, height = x2 - x1, y2 - y1
        area = width * height

        if area >= min_box_area:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            projected_clusters += 1
            print(f"[✓] Cluster {cluster_id}: {width}x{height}px → kept")
        else:
            print(f"[✗] Cluster {cluster_id}: {width}x{height}px → too small")

    print(f"[✓] Projected {projected_clusters}/{total_clusters} clusters to 2D boxes")
    return boxes

def draw_bboxes(img, boxes, color=(0, 255, 0), label='Obstacle'):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def main():
    base_dir = os.path.dirname(__file__)
    idx = 19
    image_name = f'um_0000{idx}.png'

    left_path = os.path.join(base_dir, '..', 'data_road', 'training', 'image_2', image_name)
    right_path = os.path.join(base_dir, '..', 'data_road_right', 'training', 'image_3', image_name)
    calib_path = os.path.join(base_dir, '..', 'data_road', 'training', 'calib', f'um_0000{idx}.txt')

    left_color = cv2.imread(left_path)
    left_gray = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    calib = parse_kitti_calib(calib_path)

    disparity = compute_disparity(left_gray, right_gray)
    smoothed = average_disparity_by_slic(disparity, left_color, 3000, 10.0)
    pcd = point_cloud_from_disparity(smoothed, calib, left_color, show=False)
    raw_points = np.asarray(pcd.points)

    obstacle_pts, _, plane = ransac_ground_removal(raw_points, show=False)
    filtered_pts = filter_by_height(obstacle_pts, plane, min_h=0.15, max_h=2.0)
    print(f"[✓] Filtered {len(filtered_pts)} points above ground")

    labels = cluster_obstacles_dbscan(filtered_pts, eps=0.1, min_samples=30, show=False)
    clusters = group_clusters(filtered_pts, labels)
    print(f"[✓] Clustered into {len(clusters)} obstacle groups")

    debug_img = left_color.copy()
    boxes = project_to_image(clusters, calib, left_color.shape, min_box_area=30, debug_img=debug_img)
    print(f"[✓] Final projected obstacle boxes: {len(boxes)}")

    vis = left_color.copy()
    draw_bboxes(vis, boxes)

    cv2.imshow("Stereo-Only Obstacle Detection", vis)
    cv2.imshow("Projected Points Debug", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
