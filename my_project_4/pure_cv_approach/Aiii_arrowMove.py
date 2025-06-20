from typing import List, Tuple # Python 3.7 typeError fix!
from time import time
import numpy as np
import cv2
import os

from Ai_road_finder.Ai_from_disparity import parse_kitti_calib
from Aii_object_detection.Aii_obj_detection_current import (
    YOLODetector, crop_bottom_half, overlay_mask, draw_bboxes
)

def draw_arrow_right_half(
    img:       np.ndarray,
    road_mask: np.ndarray,
    boxes:     List[Tuple[int, int, int, int]],
    line_len:  int  = -1,
    full_road: bool = False,
    step_px:   int  = 8,
    min_pts:   int  = 8,
    rj_filter: bool = True
) -> np.ndarray:
    '''
    Επιστρέφει το διάνυσμα κίνησης (κατεύθυνση), χρωματισμένο βάση
    την σύγκρουση/επαφή με εμπόδιο! Επίλεξε αν θέλεις ολόκληρο τον δρόμο.
    
    Args:
     - line_len_: Το μήκος στο οποίο θα εκτείνεται η γραμμή των dots.
     - full_road_: Σχεδιασμός ολόκληρου του δρόμου ή μόνο της δεξιάς πλευράς.
     - rj_filter_: Remove Jumps filter, ώστε να μην επηρεάζεται η
       κατεύθυνση από μεμονωμένες κουκκίδες λόγω θορύβου της μάσκας!
    '''
    # Το % των κουκκίδων που θα γίνουν κόκκινες όταν συναντήσει εμπόδιο!
    RED_FRAC = 0.3

    # Color palette
    GREEN = (0, 255, 0)
    RED   = (0, 0, 255)

    (h, _) = road_mask.shape[:2]

    centres = [] # (x, y)
    slices  = [] # (x0, x1, y) για την λεπτή γραμμή

    # Εύρεση των υποψήφων για το κέντρο του δρόμου
    # ξεκινώντας από το κάτω μέρος της εικόνας!!!
    for y in range(h - 1, line_len, -step_px):
        row_idx = np.flatnonzero(road_mask[y]) # x θέσεις όπου υπάρχει mask!
        if row_idx.size < min_pts:
            if centres: # Είχαμε ήδη κέντρο δρόμου!
                break;
            continue;

        (x_left, x_right) = (int(row_idx[0]), int(row_idx[-1]))
        if not full_road:
            x_mid     = (x_left + x_right) // 2
            right_idx = row_idx[row_idx >= x_mid]
            if right_idx.size < min_pts:
                continue;

            x_c = int((right_idx[0] + right_idx[-1]) // 2)
        else:
            right_idx = row_idx
            x_c = int((x_left + x_right) // 2)
        centres.append((x_c, y))
        slices.append((int(right_idx[0]), int(right_idx[-1]), y))

    # Δεν βρέθηκαν αρκετά κέντρα, οπότε για να αποφύγουμε πατάτα...
    if len(centres) < min_pts:
        return img;

    # Φιλτράρισμα για να αποφύγουμε το λάθος των λίγων [jumps]!
    if rj_filter:
        xs         = np.array([x for x, _ in centres])
        dominant_x = int(np.bincount(xs).argmax())
        centres    = [
            (dominant_x, y) if abs(x - dominant_x) > 6 else (x, y) \
                for (x, y) in centres
        ]

    # Αναζητούμε το 1ο slice που αγγίζει κάποιο bounding box!
    touch_idx = None
    for (i, (x0, x1, y)) in enumerate(slices):
        for (bx1, by1, bx2, by2) in boxes:
            if y >= by1 and y <= by2 and not (x1 < bx1 or x0 > bx2):
                touch_idx = i
                break;
        if touch_idx is not None:
            break;

    # Εφαρμογή του κόκκινου χρώματος βάση % αν γίνεται collision!
    if touch_idx is not None:
        red_start = int(len(centres) * (1 - RED_FRAC))
    else:
        red_start = len(centres)

    # Ζωγραφικήηηη!
    for (i, ((x, y), (x0, x1, y_slice))) in enumerate(zip(centres, slices)):
        col = RED if i >= red_start else GREEN
        cv2.circle(img, (x, y), 4, col, -1)
        cv2.line(img, (x0, y_slice), (x1, y_slice), col, 1)

    return img;

def main():
    base_dir = os.path.dirname(__file__)

    yolo_detector = YOLODetector()
    print() # Για καλύτερη εμφάνιση!

    ROAD  = (255, 0, 0) # Μπλε overlay για το δρόμο!
    image_type   = 'um'
    dataset_type = 'testing'
    dataset_type = 'training'

    calib_path = os.path.join(base_dir, f'calibration_KITTI.txt')
    calib      = parse_kitti_calib(calib_path)

    for i in range(94):
        general_name_file = (f'{image_type}_0000{i}.png' if i > 9 \
                             else f'{image_type}_00000{i}.png')
        
        left_path = os.path.join(
            base_dir, '..',
            'KITTI', 'data_road', dataset_type, 'image_2',
            general_name_file
        )
        right_path = os.path.join(
            base_dir, '..',
            'KITTI', 'data_road_right', dataset_type, 'image_3',
            general_name_file
        )

        left_color = cv2.imread(left_path)
        left_gray  = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        left_gray_cropped  = crop_bottom_half(left_gray)
        right_gray_cropped = crop_bottom_half(right_gray)

        start = time()
        (boxes, _, road_mask_cleaned) = yolo_detector.detect(
            left_color,
            left_gray_cropped, right_gray_cropped,
            calib
        )

        vis = overlay_mask(left_color, road_mask_cleaned, ROAD)
        draw_bboxes(vis, boxes)
        vis = draw_arrow_right_half(vis, road_mask_cleaned, boxes)
        
        print(f'Διάρκεια εκτέλεσης: {time() - start:.2f} sec')

        cv2.imshow('Arrow Visualization', vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
