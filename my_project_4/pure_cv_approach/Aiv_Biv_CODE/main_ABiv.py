import sys, os
base_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
)

# --- Camera Modules ---
from Aii_object_detection.Aii_obj_detection_current import (
    YOLODetector, crop_bottom_half, overlay_mask, draw_bboxes
)
from Aiii_arrowMove import (parse_kitti_calib, draw_arrow_right_half)

# --- LiDAR Modules ---
from Biii_LiDAR_arrowMove import (
    load_velodyne_bin, visualize_toggle, load_calibration
)

from time import time
import cv2

def main():
    general_name_file = 'fake'
    calib_path        = os.path.join(base_dir, '..', f'calibration_KITTI.txt')

    # =================== Camera Wall Test ===================
    yolo_detector = YOLODetector()
    calib         = parse_kitti_calib(calib_path)
    ROAD          = (255, 0, 0) # Μπλε overlay για το δρόμο!
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
        (boxes, _, road_mask_cleaned) = yolo_detector.detect(
            left_color,
            left_gray_cropped, right_gray_cropped,
            calib
        )

        # 2D αναπαράσταση
        vis = overlay_mask(left_color, road_mask_cleaned, ROAD)
        draw_bboxes(vis, boxes)
        vis = draw_arrow_right_half(vis, road_mask_cleaned, boxes)

        print(f'Διάρκεια εκτέλεσης: {time() - start:.2f} sec')

        cv2.imshow('Overlay - Camera Wall Test', vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # =================== LiDAR Visualization ===================
    (Tr_velo_to_cam, P2) = load_calibration(calib_path)
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
        visualize_toggle(
            image, points, P2, Tr_velo_to_cam,
            BEV = False
        )

    return;

if __name__ == '__main__':
    main()
