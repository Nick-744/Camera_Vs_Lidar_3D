import os
import cv2
import numpy as np
from time import time

from Ai_road_finder.Ai_from_disparity import compute_disparity, point_cloud_from_disparity, ransac_ground, ground_mask_from_points
from Aii_object_detection.stereo_obstacle_detection import draw_bboxes

def main():
    base_dir = os.path.dirname(__file__)
    calib_path = os.path.join(base_dir, 'calibration_KITTI.txt')
    calib = {
        'f': 721.5377,
        'cx': 609.5593,
        'cy': 172.854,
        'Tx': -0.532725
    }

    for idx in range(10, 12):
        name = f"um_0000{idx}.png"
        left_path = os.path.join(base_dir, '..', '..', 'KITTI', 'data_road', 'training', 'image_2', name)
        right_path = os.path.join(base_dir, '..', '..', 'KITTI', 'data_road_right', 'training', 'image_3', name)

        left_color = cv2.imread(left_path)
        left_gray = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        disparity = compute_disparity(left_gray, right_gray)
        points, mask = point_cloud_from_disparity(disparity, calib, left_color)
        obs_pts, ground_pts, plane = ransac_ground(points)
        road_mask = ground_mask_from_points(ground_pts, calib, left_color.shape)

        # Estimate direction
        road_blur = cv2.GaussianBlur(road_mask.astype(np.uint8) * 255, (21, 21), 0)
        M = cv2.moments(road_blur)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.arrowedLine(left_color, (cx, cy), (cx, cy - 80), (255, 255, 0), 3)
            if len(obs_pts) > 0:
                cv2.circle(left_color, (cx, cy), 20, (0, 0, 255), 3)

        cv2.imshow("Aiii - Direction + Obstacle", left_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
