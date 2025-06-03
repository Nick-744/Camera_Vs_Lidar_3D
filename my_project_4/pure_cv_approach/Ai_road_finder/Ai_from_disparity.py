from time import time
import open3d as o3d
import numpy as np
import cv2
import os

# --- Απαιτούμενες πληροφορίες για το setup του KITTI ---
def parse_kitti_calib(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    (P2, P3) = (None, None)
    for line in lines:
        if line.startswith('P2:'):
            P2 = np.array(
                list(map(float, line.strip().split()[1:]))
            ).reshape(3, 4)
        elif line.startswith('P3:'):
            P3 = np.array(
                list(map(float, line.strip().split()[1:]))
            ).reshape(3, 4)

    f  = P2[0, 0]
    cx = P2[0, 2]
    cy = P2[1, 2]
    Tx = -(P3[0, 3] - P2[0, 3]) / f

    return {'f': f, 'cx': cx, 'cy': cy, 'Tx': Tx};

# --- Υπολογισμός μάσκας / Εύρεση δρόμου ---
def compute_disparity(left_gray:      np.ndarray,
                      right_gray:     np.ndarray,
                      numDisparities: int = 128,
                      show:           bool = False) -> np.ndarray:
    ''' - numDisparities: Πρέπει να είναι πολλαπλάσιο του 16! '''
    # https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
    block_size = 5
    block_size_squared = block_size * block_size

    stereo = cv2.StereoSGBM_create(
        minDisparity      = 0,
        numDisparities    = numDisparities,
        blockSize         = block_size,
        P1                = 8  * 3 * block_size_squared,
        P2                = 32 * 3 * block_size_squared,
        disp12MaxDiff     = 1,
        uniquenessRatio   = 5,
        speckleWindowSize = 100, # Βοηθάει με τον θόρυβο!
        speckleRange      = 2,
        preFilterCap      = 63,
        mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # r / 16., λόγω του εσωτερικού τρόπου υπολογισμού [πράξεις με int]!
    disparity = stereo.compute(
        left_gray, right_gray
    ).astype(np.float32) / 16.;

    if show:
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)
        cv2.imshow('Disparity Map - Debug', disp_vis)
        cv2.waitKey(1)

    return disparity;

def point_cloud_from_disparity(disparity: np.ndarray,
                               calib:     dict) -> np.ndarray:
    '''
    disparity + calib -> 3D point cloud!
    '''
    (f, cx, cy, Tx) = (
        calib['f'], calib['cx'], calib['cy'], calib['Tx']
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

    return points;

def ransac_ground(points:             np.ndarray,
                  distance_threshold: float = 0.02,
                  ransac_n:           int = 3,
                  num_iterations:     int = 3000,
                  show:               bool = False) -> tuple:
    '''
    Υπολογισμός/Εύρεση επιπέδου δρόμου με RANSAC.

    Returns:
        out: tuple
        - Σημεία που ανήκουν σε εμπόδια [obstacle_points]
        - Σημεία που ανήκουν στο έδαφος [ ground_points ]
        - Συντεταγμένες του επιπέδου    [  plane_model  ]
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    (plane_model, inliers) = pcd.segment_plane(
        distance_threshold, ransac_n, num_iterations
    )

    obstacle_points = pcd.select_by_index(inliers, invert = True)
    ground_points = pcd.select_by_index(inliers)
    if show:
        obstacle_points.paint_uniform_color([1., 0., 0.]) # Κόκκινο
        ground_points.paint_uniform_color(  [0., 1., 0.]) # Πράσινο
        o3d.visualization.draw_geometries([ground_points, obstacle_points])

    return (
        np.asarray(obstacle_points.points),
        np.asarray(ground_points.points),
        plane_model
    );

def project_points_to_mask(points:      np.ndarray,
                           calib:       dict,
                           shape:       tuple,
                           crop_bottom: bool = True) -> np.ndarray:
    '''
    Προβολή 3D σημείων σε μάσκα 2D
    '''
    (f, cx, cy) = (calib['f'], calib['cx'], calib['cy'])
    (h, w) = shape[:2]

    v_offset = h // 2 if crop_bottom else 0

    (X, Y, Z) = (points[:, 0], points[:, 1], points[:, 2])
    valid = (Z > 0.1) & (Z < 40)
    (X, Y, Z) = (X[valid], Y[valid], Z[valid])

    u = np.round((f * X / Z) + cx).astype(np.int32)
    v = np.round((f * Y / Z) + cy).astype(np.int32) + v_offset

    valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    (u, v) = (u[valid_uv], v[valid_uv])

    mask = np.zeros((h, w), dtype = np.uint8)
    mask[v, u] = 255

    return mask;

def detect_ground_mask(left_gray:        np.ndarray,
                       right_gray:       np.ndarray,
                       original_shape:   tuple,
                       calib:            dict,
                       numDisparities:   int = 128,
                       ransac_threshold: float = 0.02,
                       crop_bottom:      bool = True,
                       debug:            bool = False) -> np.ndarray:
    '''
    Εξαγωγή μάσκας δρόμου από disparity map μέσω RANSAC.
    '''
    disparity = compute_disparity(
        left_gray, right_gray, numDisparities = numDisparities,
        show = debug
    )
    points = point_cloud_from_disparity(disparity, calib)
    (_, ground_pts, _) = ransac_ground(
        points, distance_threshold = ransac_threshold,
        show = debug
    )

    return project_points_to_mask(
        ground_pts, calib, original_shape, crop_bottom
    );

# --- Helpers ---
def crop_bottom_half(image: np.ndarray) -> np.ndarray:
    h = image.shape[0]

    return image[h // 2:, :];

def overlay_mask(image: np.ndarray,
                 mask:  np.ndarray,
                 color: tuple = (0, 0, 255),
                 alpha: float = 0.5) -> np.ndarray:
    ''' Προβολή διαφανούς μάσκας σε εικόνα '''
    # Οι δείκτες των pixels μάσκας που είναι foreground
    idx = mask.astype(bool)

    # Δημιουργία solid χρώματος για την μάσκα
    solid = np.empty_like(image[idx])
    solid[:] = color

    # Συνδυασμός της αρχικής εικόνας με το solid χρώμα (mask)
    image[idx] = cv2.addWeighted(image[idx], 1 - alpha, solid, alpha, 0)

    return image;

def post_process_mask(mask:        np.ndarray,
                      min_area:    int = 5000,
                      kernel_size: int = 7) -> np.ndarray:
    '''
    Επεξεργασία μάσκας για καλύτερη εμφάνιση!
    '''
    # Binary
    mask = (mask > 0).astype(np.uint8) * 255

    # Morphological closing (για να κλείσουν τα κενά)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Morphological opening (για να αφαιρεθούν μικρές περιοχές)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Αφαίρεση μικρών περιοχών
    (num_labels, labels, stats, _) = cv2.connectedComponentsWithStats(mask)
    cleaned_mask = np.zeros_like(mask)

    for i in range(1, num_labels): # skip bg (i = 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255

    cleaned_mask = cv2.GaussianBlur(cleaned_mask, (11, 11), 0)

    return cleaned_mask;

def main():
    base_dir = os.path.dirname(__file__)
    
    image_type   = 'um'
    dataset_type = 'testing'
    dataset_type = 'training'

    calib_path = os.path.join(base_dir, '..', f'calibration_KITTI.txt')
    calib      = parse_kitti_calib(calib_path)

    for i in range(94):
        general_name_file = (f'{image_type}_0000{i}.png' if i > 9 \
                             else f'{image_type}_00000{i}.png')
        
        left_path = os.path.join(
            base_dir, '..', '..',
            'KITTI', 'data_road', dataset_type, 'image_2',
            general_name_file
        )
        right_path = os.path.join(
            base_dir, '..', '..',
            'KITTI', 'data_road_right', dataset_type, 'image_3',
            general_name_file
        )

        left_color = cv2.imread(left_path)
        left_gray  = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        # Αφαίρεση του άνω μισού της εικόνας (γρηγορότερο disparity!)
        left_gray_cropped  = crop_bottom_half(left_gray)
        right_gray_cropped = crop_bottom_half(right_gray)

        # Βασική εκτέλεση
        start = time()
        mask = detect_ground_mask(
            left_gray_cropped,
            right_gray_cropped,
            left_color.shape,
            calib,
            crop_bottom = True
        )
        print(f'Διάρκεια εκτέλεσης: {time() - start:.2f} sec')

        # min_area = 15000 <-> 20000 καλύτερα αποτελέσματα!
        mask_cleaned = post_process_mask(
            mask, min_area = 15000, kernel_size = 7
        ) # Post-processing του mask για καλύτερη εμφάνιση!
        result = overlay_mask(left_color, mask_cleaned)

        cv2.imshow('Overlay', result)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
