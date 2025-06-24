import os
import cv2
import numpy as np
import open3d as o3d
from time import time

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
        pcd, Tr_velo_to_cam, P2
    )
    pts_kept = pcd[mask_depth]

    # Φιλτράρουμε τα σημεία που είναι της κάμερας (Field of View)
    (h, w) = image_shape[:2]
    fov_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    return pts_kept[fov_mask];

def filter_by_height(points: np.ndarray,
                     plane:  tuple,
                     min_h:  float = -0.2,
                     max_h:  float = 0.2) -> np.ndarray:
    ''' Κρατά σημεία σε υψομετρικό εύρος από το επίπεδο '''
    a, b, c, d = plane
    dist = (points @ np.array([a, b, c]) + d) / np.linalg.norm([a, b, c])
    mask = (dist > min_h) & (dist < max_h)
    
    return points[mask];

'''
Άλλα φίλτρα που δοκιμάστηκαν, αλλά απλά έκαναν πιο αργή την εκτέλεση,
χωρίς σημαντική βελτίωση στην ποιότητα εύρεσης του δρόμου...

def filter_points_near_plane(...):
    Κρατούσε τα σημεία που βρίσκονταν κοντά στο επίπεδο/δρόμο!


def filter_points_by_local_smoothness(...)
    Αφαιρούσε σημεία με απότομες διαφορές ύψους με τους γείτονές τους.
'''

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

    (plane_model, inliers) = pcd.segment_plane(
        distance_threshold = distance_threshold,
        ransac_n           = ransac_n,
        num_iterations     = num_iterations
    )

    ground     = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert = True)
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
        pcd, Tr_velo_to_cam, P2
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
        mask, connectivity = 8
    )
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_binary = (labels == largest).astype(np.uint8)
    
    return mask_binary * 255;

def my_road_from_pcd_is(pcd:            np.ndarray,
                        Tr_velo_to_cam: np.ndarray,
                        P2:             np.ndarray,
                        image_shape:    tuple,
                        debug:          bool = False) -> tuple:
    '''
    Επιστρέφει τη μάσκα του δρόμου που βρήκε από το pcd/LiDAR,
    μαζί με τα σημεία του δρόμου και το επίπεδο του δρόμου.

    Params:
     - Tr_velo_to_cam : Ο μετασχηματισμός από το σύστημα αναφοράς
                        του LiDAR στην κάμερα.
     - P2             : Ο πίνακας προβολής της κάμερας.
     - filter         : Τελικά δεν υλοποιήθηκε λόγω αύξησης του
                        χρόνου εκτέλεσης!
    '''
    visible_points = filter_visible_points(
        pcd, Tr_velo_to_cam, P2, image_shape
    )
    if debug:
        print(f'Ορατά σημεία: {visible_points.shape[0]}')
    
    (ground_points, plane) = detect_ground_plane(
        visible_points,
        distance_threshold = 0.025,
        num_iterations     = 20000,
        show = debug
    )

    road_mask = project_points_to_image(
        ground_points, Tr_velo_to_cam, P2, image_shape
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

def overlay_mask(image: np.ndarray,
                 mask:  np.ndarray,
                 color: tuple = (0, 0, 255),
                 alpha: float = 0.8) -> np.ndarray:
    ''' Προβολή διαφανούς μάσκας σε εικόνα '''
    # Οι δείκτες των pixels μάσκας που είναι foreground
    idx = mask.astype(bool)

    # Δημιουργία solid χρώματος για την μάσκα
    solid = np.empty_like(image[idx])
    solid[:] = color

    # Συνδυασμός της αρχικής εικόνας με το solid χρώμα (mask)
    image[idx] = cv2.addWeighted(image[idx], 1 - alpha, solid, alpha, 0)

    return image;

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
                (key, value) = line.strip().split(':', 1)
                calib[key] = np.array(
                    [float(x) for x in value.strip().split()]
                )

    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])

    P2 = calib['P2'].reshape(3, 4)

    return (Tr_velo_to_cam, P2);

def main():
    base_dir = os.path.dirname(__file__)

    image_type = 'um'
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
        
        image  = cv2.imread(img_path)
        points = load_velodyne_bin(bin_path)

        start = time()
        (road_mask, _, _) = my_road_from_pcd_is(
            points, Tr_velo_to_cam, P2, image.shape
        )
        print(f'Διάρκεια εκτέλεσης: {time() - start:.2f} sec / {general_name_file}')

        # Ζωγραφικηηή!
        overlay = overlay_mask(
            image, road_mask, color = (255, 0, 0), alpha = 0.5
        )

        cv2.imshow('LiDAR ROAD', overlay)
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
