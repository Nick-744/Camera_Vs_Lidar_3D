import os
import cv2
import numpy as np
import open3d as o3d

def load_velodyne_bin(bin_path: str) -> np.ndarray:
    """
    Load a KITTI .bin Velodyne point cloud file (Nx4) → return Nx3 (XYZ).
    """
    points = np.fromfile(bin_path, dtype = np.float32).reshape(-1, 4)

    return points[:, :3];

def load_calibration(calib_path: str) -> tuple:
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

def project_points_to_image(points, Tr_velo_to_cam, P2, image_shape):
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    cam_points = Tr_velo_to_cam @ points_hom.T
    cam_points = cam_points[:3, :]

    valid = cam_points[2, :] > 0.1
    cam_points = cam_points[:, valid]

    pixels = P2 @ np.vstack([cam_points, np.ones((1, cam_points.shape[1]))])
    pixels /= pixels[2, :]

    u = np.round(pixels[0, :]).astype(int)
    v = np.round(pixels[1, :]).astype(int)

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask[v[valid], u[valid]] = 1

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Keep largest connected component AFTER dilation
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    # mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
    
    return (mask > 0.1).astype(np.uint8) * 255;

def detect_ground_plane(points: np.ndarray,
                        distance_threshold: float = 0.2,
                        ransac_n: int = 3,
                        num_iterations: int = 1000,
                        show: bool = False) -> tuple:
    """
    Detect the road (ground) using RANSAC plane fitting.
    """
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

def main():
    base_dir = os.path.dirname(__file__)

    for i in range(10, 50):
        general_name_file = f'um_0000{i}'
        bin_path = os.path.join(
            base_dir,
            '..',
            'KITTI',
            'data_road_velodyne',
            'training',
            'velodyne',
            f'{general_name_file}.bin'
        )
        img_path = os.path.join(
            base_dir, 
            '..',
            'KITTI',
            'data_road',
            'training',
            'image_2',
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
        if (not os.path.exists(bin_path)) or \
            (not os.path.exists(img_path)) or \
            (not os.path.exists(calib_path)):
            print(f'Πρόβλημα με το {general_name_file}')
            continue;
        
        Tr_velo_to_cam, P2 = load_calibration(calib_path)

        points = load_velodyne_bin(bin_path)
        image = cv2.imread(img_path)
        (ground_points, _) = detect_ground_plane(
            points,
            distance_threshold = 0.05
        )

        road_mask = project_points_to_image(
            ground_points,
            Tr_velo_to_cam,
            P2,
            image.shape
        )

        # Ζωγραφικηηή!
        green_mask = np.zeros_like(image)
        green_mask[road_mask == 255] = [0, 255, 0]
        overlay = cv2.addWeighted(image, 0.7, green_mask, 0.3, 0)

        cv2.imshow("Road Mask Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return;

if __name__ == "__main__":
    main()
