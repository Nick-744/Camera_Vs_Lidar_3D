import os
import cv2
import numpy as np
import open3d as o3d
from time import time
from scipy.spatial import cKDTree

def filter_visible_points(pcd:            np.ndarray,
                          Tr_velo_to_cam: np.ndarray,
                          P2:             np.ndarray,
                          image_shape:    tuple) -> np.ndarray:
    '''
    Φιλτράρει το pcd έτσι ώστε να επεξεργαστούμε μόνο τα
    ορατά από την κάμερα σημεία. Επιστρέφει το ορατό pcd.

    Witcher 3 type of optimization sh*t!
    '''
    (_, u, v, mask_depth) = project_lidar_to_image(
        pcd,
        Tr_velo_to_cam,
        P2
    )
    pts_kept = pcd[mask_depth]

    # Φιλτράρουμε τα σημεία που είναι της κάμερας (Field of View)
    (h, w) = image_shape[:2]
    fov_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    return pts_kept[fov_mask];

def detect_ground_plane(points:             np.ndarray,
                        distance_threshold: float = 0.2,
                        ransac_n:           int = 3,
                        num_iterations:     int = 1000,
                        show:               bool = False) -> tuple:
    '''
    Εύρεση του εδάφους/δρόμου του point cloud με RANSAC.

    Επιστρέφει τα σημεία του δρόμου και 1 tuple (a, b, c, d)
    που ορίζει το επίπεδο που βρέθηκε με RANSAC.
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold = distance_threshold,
        ransac_n           = ransac_n,
        num_iterations     = num_iterations
    )

    ground = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert=True)
    if show:
        ground.paint_uniform_color([0.0, 1.0, 0.0])
        non_ground.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries([ground, non_ground])

    return (np.asarray(ground.points), plane_model);

def project_points_to_image(pcd:            np.ndarray,
                            Tr_velo_to_cam: np.ndarray,
                            P2:             np.ndarray,
                            image_shape:    tuple) -> np.ndarray:
    '''
    Προβολή του pcd στην εικόνα με χρήση του camera
    projection matrix P2. Ίδια λογική λειτουργίας με το
    filter_visible_points, απλά τώρα επιστρέφει μάσκα!
    '''
    (_, u, v, _) = project_lidar_to_image(
        pcd,
        Tr_velo_to_cam,
        P2
    )

    (h, w) = image_shape[:2]
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # Δημιουργία μάσκας:
    mask = np.zeros((h, w), dtype = np.uint8)
    mask[v[valid], u[valid]] = 1

    # Morphological επεξεργασία για βελτίωση της μάσκας!
    kernel = np.ones((6, 6), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    (num_labels, labels, stats, _) = cv2.connectedComponentsWithStats(
        mask,
        connectivity = 8
    )
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_binary = (labels == largest).astype(np.uint8)
    
    return mask_binary * 255;

def filter_points_by_local_smoothness(
    points:        np.ndarray,
    radius:        float = 0.3,
    height_thresh: float = 0.05
) -> np.ndarray:
    '''
    Αφαιρεί σημεία με απότομες διαφορές ύψους με τους γείτονές τους.

    - radius:
        Ακτίνα ελέγχου (σε μέτρα).
    - height_thresh:
        Μέγιστη επιτρεπτή τυπική απόκλιση στο Z των γειτόνων.
    '''
    kdtree = cKDTree(points[:, :2])
    smooth_points = []

    for (i, pt) in enumerate(points):
        idxs = kdtree.query_ball_point(pt[:2], radius)
        if len(idxs) < 3:
            continue;
        z_vals = points[idxs][:, 2]
        if np.std(z_vals) < height_thresh:
            smooth_points.append(pt)

    return np.array(smooth_points);

def filter_points_near_plane(points:      np.ndarray,
                             plane_model: tuple,
                             max_dist:    float = 0.15) -> np.ndarray:
    '''
    Κρατάει τα σημεία που βρίσκονται κοντά στο επίπεδο/δρόμο!

    - plane_model:
        Το επίπεδο (a, b, c, d) από RANSAC: ax + by + cz + d = 0.
    - max_dist:
        Μέγιστη απόσταση για να θεωρηθεί ότι ανήκει στο επίπεδο.
    '''
    (a, b, c, d) = plane_model
    normal = np.array([a, b, c])
    distances = np.abs(points @ normal + d) / np.linalg.norm(normal)
    
    return points[distances < max_dist];

def my_road_from_pcd_is(pcd:            np.ndarray,
                        Tr_velo_to_cam: np.ndarray,
                        P2:             np.ndarray,
                        image_shape:    tuple,
                        debug:          bool = False,
                        apply_filters:  bool = False) -> tuple:
    '''
    Επιστρέφει τη μάσκα του δρόμου που βρήκε από το pcd/LiDAR,
    μαζί με τα σημεία του δρόμου και το επίπεδο του δρόμου.
    '''
    visible_points = filter_visible_points(
        pcd,
        Tr_velo_to_cam,
        P2,
        image_shape
    )
    if debug:
        print(f'Ορατά σημεία: {visible_points.shape[0]}')
    
    (ground_points, plane) = detect_ground_plane(
        visible_points,
        distance_threshold = 0.02,
        num_iterations = 10000,
        show = debug
    )

    # Φίλτρα:
    if apply_filters:
        ground_points = filter_points_near_plane(
            ground_points,
            plane,
            max_dist = 0.02
        )
        ground_points = filter_points_by_local_smoothness(
            ground_points,
            radius = 0.08,
            height_thresh = 0.02
        )

    road_mask = project_points_to_image(
        ground_points,
        Tr_velo_to_cam,
        P2,
        image_shape
    )

    return (road_mask, ground_points, plane);

# ----- Helpers -----
def project_lidar_to_image(points:         np.ndarray,
                           Tr_velo_to_cam: np.ndarray,
                           P2:             np.ndarray) -> tuple:
    ''' Μετατρέπει 3D σημεία (pcd) από LiDAR -> camera -> pixels! '''
    # LiDAR -> camera frame (με χρήση Homogeneous coordinates)
    pts_h = np.c_[points, np.ones(points.shape[0])] # N x 4
    cam = (Tr_velo_to_cam @ pts_h.T)[:3]            # 3 επειδή x, y, z
    depth_mask = cam[2] > 0.1
    cam = cam[:, depth_mask] # Πολύ κοντά στην κάμερα

    # Camera frame -> pixels
    '''
    P2: 3 x 4 camera projection matrix του KITTI calibration,
    που μετατρέπει 3D camera coords σε 2D image pixels!
    '''
    pixels = P2 @ np.vstack([cam, np.ones(cam.shape[1])])
    pixels /= pixels[2] # Από homogeneous σε Cartesian coordinates

    u = np.round(pixels[0]).astype(int)
    v = np.round(pixels[1]).astype(int)

    return (cam, u, v, depth_mask);

# ----- I/O -----
def load_velodyne_bin(bin_path: str) -> np.ndarray:
    '''
    Φορτώνει το Velodyne pcd .bin αρχείο (του KITTI)
    και επιστρέφει τα σημεία του.
    '''
    points = np.fromfile(bin_path, dtype = np.float32).reshape(-1, 4)

    return points[:, :3];

def load_calibration(calib_path: str) -> tuple:
    '''
    Φορτώνει το calibration .txt αρχείο του KITTI
    και επιστρέφει το Tr_velo_to_cam και P2.
    '''
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                calib[key] = np.array(
                    [float(x) for x in value.strip().split()]
                )

    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])

    P2 = calib['P2'].reshape(3, 4)

    return (Tr_velo_to_cam, P2);

def main():
    base_dir = os.path.dirname(__file__)

    for i in range(10, 90):
        general_name_file = f'um_0000{i}'
        bin_path = os.path.join(
            base_dir,
            '..', '..',
            'KITTI',
            'data_road_velodyne',
            'training',
            'velodyne',
            f'{general_name_file}.bin'
        )
        img_path = os.path.join(
            base_dir, 
            '..', '..',
            'KITTI',
            'data_road',
            'training',
            'image_2',
            f'{general_name_file}.png'
        )
        calib_path = os.path.join(
            base_dir,
            '..', '..',
            'KITTI',
            'data_road',
            'training',
            'calib',
            f'{general_name_file}.txt'
        )
        if not (os.path.isfile(bin_path) and \
                os.path.isfile(img_path) and \
                os.path.isfile(calib_path)):
            print(f'Πρόβλημα με το {general_name_file}')
            continue;
        
        (Tr_velo_to_cam, P2) = load_calibration(calib_path)
        image  = cv2.imread(img_path)
        points = load_velodyne_bin(bin_path)

        start = time()
        (road_mask, _, _) = my_road_from_pcd_is(
            points,
            Tr_velo_to_cam,
            P2,
            image.shape,
            apply_filters = True
        )
        print(f'Διάρκεια εκτέλεσης: {time() - start:.2f} sec')

        # Ζωγραφικηηή!
        mask_colored = np.zeros_like(image)
        mask_colored[road_mask == 255] = [255, 0, 0]
        overlay = cv2.addWeighted(image, 0.6, mask_colored, 0.4, 0)

        cv2.imshow("Μάσκα από LiDAR", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return;

if __name__ == "__main__":
    main()
