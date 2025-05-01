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

def main():
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

if __name__ == "__main__":
    main()
