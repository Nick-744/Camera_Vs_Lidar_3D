import sys, os
base_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
)

# --- Camera Modules ---
from Aii_object_detection.Aii_obj_detection_current import (
    YOLODetector, crop_bottom_half, overlay_mask,
    draw_bboxes, detect_obstacles
)
from Aiii_arrowMove import (parse_kitti_calib, draw_arrow_right_half)

# --- LiDAR Modules ---
from Biii_LiDAR_arrowMove import (
    load_velodyne_bin, visualize_toggle, load_calibration
)

from time import time
import numpy as np
import cv2

# ========================= Camera Wall Test =========================
def test_camera_wall(calib_path:        str,
                     general_name_file: str,
                     use_yolo:          bool = True) -> None:
    if use_yolo:
        yolo_detector = YOLODetector()
        
    calib         = parse_kitti_calib(calib_path)
    ROAD          = (255, 0, 0) # Μπλε overlay για το δρόμο!

    print('\n-> Camera Wall Test')
    for i in range(1, 4):
        left_path  = os.path.join(
            base_dir, 'DATA', f'{general_name_file}_left_{i}.png'
        )
        right_path = os.path.join(
            base_dir, 'DATA', f'{general_name_file}_right_{i}.png'
        )

        left_color = cv2.imread(left_path)
        left_gray  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        # Αφαίρεση του άνω μισού της εικόνας (γρηγορότερο disparity!)
        left_gray_cropped  = crop_bottom_half(left_gray)
        right_gray_cropped = crop_bottom_half(right_gray)

        # Βασική εκτέλεση
        start = time()
        if use_yolo:
            (boxes, _, road_mask_cleaned) = yolo_detector.detect(
                left_color,
                left_gray_cropped, right_gray_cropped,
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

        # 2D αναπαράσταση
        vis = overlay_mask(left_color, road_mask_cleaned, ROAD)
        draw_bboxes(vis, boxes)
        vis = draw_arrow_right_half(
            vis, road_mask_cleaned, boxes,
            full_road = not (True * (i % 2)),
            rj_filter = True * (i % 2)
        )

        print(f'Διάρκεια εκτέλεσης: {time() - start:.2f} sec')

        cv2.imshow('Overlay - Camera Wall Test', vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return;

# ========================= LiDAR Wall Test =========================
def test_lidar_wall(calib_path: str, general_name_file: str) -> None:
    (Tr_velo_to_cam, P2) = load_calibration(calib_path)

    walls = [
        #   x_range,   y_range,   z_range
        (12.7, 12.8,   -5, 5.4, -1.6, 1.4),
        (11.5, 11.6, -5.3, 5.5, -1.8, 1.4),
        (  16, 16.1, -5.6, 5.2, -1.6, 1.6)
    ]

    print('\n-> LiDAR Wall Test')
    for i in range(1, 4):
        bin_path = os.path.join(
            base_dir, 'DATA', f'{general_name_file}_{i}.bin'
        )
        img_path = os.path.join(
            base_dir, 'DATA', f'{general_name_file}_left_{i}.png'
        )

        # 3D αναπαράσταση
        image  = cv2.imread(img_path)
        points = load_velodyne_bin(bin_path)
        points = add_synthetic_wall(
            points,
            x_range = walls[i-1][ :2],
            y_range = walls[i-1][2:4],
            z_range = walls[i-1][4: ],
            density = 0.1
        )
        visualize_toggle(
            image, points, P2, Tr_velo_to_cam,
            # BEV = True
        )
    
    return;

# --- Helpers ---
def add_synthetic_wall(points:  np.ndarray,
                       x_range: tuple,
                       y_range: tuple,
                       z_range: tuple,
                       density: float = 0.2) -> np.ndarray:
    '''
    Προσθήκη ενός τεχνητού τοίχου στο PCD του LiDAR.

    Parameters:
        density: Η απόσταση μεταξύ των σημείων του τοίχου.

    Returns:
        Σύνθεση των αρχικών σημείων με τα σημεία του τοίχου.
    '''
    x_vals = np.arange(x_range[0], x_range[1] + density, density)
    y_vals = np.arange(y_range[0], y_range[1] + density, density)
    z_vals = np.arange(z_range[0], z_range[1] + density, density)
    wall   = np.array(np.meshgrid(x_vals, y_vals, z_vals)).reshape(3, -1).T

    return np.vstack((points, wall));



def main():
    general_name_file = 'fake'
    calib_path        = os.path.join(base_dir, '..', f'calibration_KITTI.txt')

    test_camera_wall(calib_path, general_name_file)
    test_lidar_wall(calib_path, general_name_file)

    return;

if __name__ == '__main__':
    main()
