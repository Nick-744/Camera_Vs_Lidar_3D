import os
import cv2
import numpy as np
import open3d as o3d
from time import time

def parse_kitti_calib(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    (P2, P3) = (None, None)
    for line in lines:
        if line.startswith("P2:"):
            P2 = np.array(
                list(map(float, line.strip().split()[1:]))
            ).reshape(3, 4)
        elif line.startswith("P3:"):
            P3 = np.array(
                list(map(float, line.strip().split()[1:]))
            ).reshape(3, 4)

    f = P2[0, 0]
    cx = P2[0, 2]
    cy = P2[1, 2]
    Tx = -(P3[0, 3] - P2[0, 3]) / f

    return {"f": f, "cx": cx, "cy": cy, "Tx": Tx};

def compute_disparity(left_gray, right_gray):
    stereo = cv2.StereoSGBM_create(
        minDisparity      = 0,
        numDisparities    = 96,
        blockSize         = 9,
        P1                = 8 * 3 * 9 * 9,
        P2                = 32 * 3 * 9 * 9,
        disp12MaxDiff     = 1,
        uniquenessRatio   = 10,
        speckleWindowSize = 50,
        speckleRange      = 2,
        preFilterCap      = 63,
        mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    return stereo.compute(
        left_gray,
        right_gray
    ).astype(np.float32) / 16.0;

def point_cloud_from_disparity(disparity, calib, left_color):
    (f, cx, cy, Tx) = (
        calib['f'],
        calib['cx'],
        calib['cy'],
        calib['Tx']
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

def ransac_ground(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    (plane_model, inliers) = pcd.segment_plane(
        distance_threshold = 0.02,
        ransac_n = 3,
        num_iterations = 2000
    )
    ground_points = pcd.select_by_index(inliers)

    return np.asarray(ground_points.points);

def project_points_to_mask(points, calib, shape):
    (f, cx, cy) = (calib['f'], calib['cx'], calib['cy'])
    (h, w) = shape[:2]
    mask = np.zeros((h, w), dtype = np.uint8)
    for (x, y, z) in points:
        if not (0.1 < z < 40):
            continue;
        u = int(round((f * x / z) + cx))
        v = int(round((f * y / z) + cy))
        if (0 <= u < w) and (0 <= v < h):
            mask[v, u] = 255
    
    return mask;

def main():
    base_dir = os.path.dirname(__file__)

    for idx in range(10, 30):
        general_name_file = f"um_0000{idx}"
        left_path = os.path.join(
            base_dir,
            '..',
            'KITTI',
            'data_road',
            'training',
            'image_2',
            f'{general_name_file}.png'
        )
        right_path = os.path.join(
            base_dir,
            '..',
            'KITTI',
            'data_road_right',
            'training',
            'image_3',
            f'{general_name_file}.png'
        )
        calib_path = os.path.join(
            base_dir,
            '..',
            'KITTI',
            'data_road',
            'training',
            'calib',
            f'{general_name_file}.txt'
        )

        left_color = cv2.imread(left_path)
        left_gray =  cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        calib = parse_kitti_calib(calib_path)

        # Κύρια διαδικασία:
        start = time()
        disparity = compute_disparity(left_gray, right_gray)
        points = point_cloud_from_disparity(disparity, calib, left_color)
        ground_pts = ransac_ground(points)
        mask = project_points_to_mask(ground_pts, calib, left_color.shape)
        print(f'Διάρκεια εκτέλεσης: {time() - start:.2f} sec')

        left_color[mask == 255] = [255, 0, 0]

        #cv2.imshow("Detected Road Mask", mask)
        cv2.imshow("Overlay", left_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
