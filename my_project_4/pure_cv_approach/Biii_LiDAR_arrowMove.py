import os
import cv2
import numpy as np
import open3d as o3d
from time import time

from part_B.Bii_LiDAR_obstacle_detect import (
    detect_obstacles_withLiDAR,
    project_clusters_with_dots
)
from part_B.Bi_road_detection_pcd import (
    load_velodyne_bin,
    load_calibration,
    project_lidar_to_image
)

def prepare_processed_pcd(image:          np.ndarray,
                          points:         np.ndarray,
                          P2:             np.ndarray,
                          Tr_velo_to_cam: np.ndarray) -> o3d.geometry.PointCloud:
    (_, clusters, road_points) = detect_obstacles_withLiDAR(
        image, points, P2, Tr_velo_to_cam
    )

    # --- Δρόμος ---
    road_pcd        = o3d.geometry.PointCloud()
    road_pcd.points = o3d.utility.Vector3dVector(road_points)
    road_pcd.paint_uniform_color([0., 0., 1.]) # Μπλε δρόμος

    # --- Εμπόδια ---
    dot_colors = { # Λεξικό/HashMap: cluster_id -> color
        dot['cluster_id']: (np.array(dot['color'], dtype = np.float32)[::-1] / 255.)
        for dot in project_clusters_with_dots(clusters, Tr_velo_to_cam, P2, image.shape)
    }

    # Pre-allocate τον max χώρο που μπορεί να χρειαστεί!
    N_total    = sum(len(pts) for pts in clusters.values())
    obs_points = np.empty((N_total, 3), dtype = np.float32)
    obs_colors = np.empty((N_total, 3), dtype = np.float32)

    i = 0
    for (cluster_id, cluster_pts) in clusters.items():
        color = dot_colors.get(cluster_id)
        if color is None:
            continue;
        n = len(cluster_pts)
        obs_points[i:i+n] = cluster_pts
        obs_colors[i:i+n] = color # Broadcasted
        i += n

    # Trim
    obs_points = obs_points[:i]
    obs_colors = obs_colors[:i]

    obs_pcd        = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(obs_points)
    obs_pcd.colors = o3d.utility.Vector3dVector(obs_colors)

    return road_pcd + obs_pcd;

def project_colors_on_pcd(image:          np.ndarray,
                          points:         np.ndarray,
                          P2:             np.ndarray,
                          Tr_velo_to_cam: np.ndarray) -> o3d.geometry.PointCloud:
    (_, u, v, mask) = project_lidar_to_image(points, Tr_velo_to_cam, P2)

    (h, w) = image.shape[:2]
    valid  = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    (u, v) = (u[valid], v[valid])

    rgb_colors   = image[v, u, ::-1] / 255. # BGR to RGB
    valid_points = points[mask][valid]      # LiDAR-space points

    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    return pcd;

TOGGLE_STATE = {'show_raw': False}
def visualize_toggle(image:          np.ndarray,
                     points:         np.ndarray,
                     P2:             np.ndarray,
                     Tr_velo_to_cam: np.ndarray) -> None:
    start = time()
    temp = (image, points, P2, Tr_velo_to_cam)
    raw_rgb_pcd   = project_colors_on_pcd(*temp)
    processed_pcd = prepare_processed_pcd(*temp)
    print(f'Χρόνος εκτέλεσης: {time() - start:.2f} sec')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width = 1000, height = 600)
    vis.add_geometry(processed_pcd)

    # --- Camera Car Viewpoint Setup ---
    # Θέτουμε την κάμερα Open3D να ταιριάζει με την κάμερα KITTI!
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()

    # Εφαρμογή του μετασχηματισμού LiDAR -> Camera!
    params.extrinsic = Tr_velo_to_cam # 4x4 matrix: LiDAR -> Camera
    view_control.convert_from_pinhole_camera_parameters(params)

    def _toggle_callback(vis_obj):
        TOGGLE_STATE['show_raw'] = not TOGGLE_STATE['show_raw']
        view = vis_obj.get_view_control()
        cam  = view.convert_to_pinhole_camera_parameters()

        vis_obj.clear_geometries()
        vis_obj.add_geometry(raw_rgb_pcd if TOGGLE_STATE['show_raw'] \
                             else processed_pcd)
        view.convert_from_pinhole_camera_parameters(cam)

        return False;

    vis.register_key_callback(ord(' '), _toggle_callback)
    vis.run()
    vis.destroy_window()

    return;

def main():
    base_dir = os.path.dirname(__file__)

    image_type   = 'um'
    dataset_type = 'testing'
    dataset_type = 'training'

    calib_path = os.path.join(base_dir, 'calibration_KITTI.txt')
    (Tr_velo_to_cam, P2) = load_calibration(calib_path)
    
    for i in range(94):
        general_name_file = (f'{image_type}_0000{i}' if i > 9 \
                             else f'{image_type}_00000{i}')

        bin_path = os.path.join(
            base_dir, '..',
            'KITTI', 'data_road_velodyne', dataset_type, 'velodyne',
            f'{general_name_file}.bin'
        )
        img_path = os.path.join(
            base_dir, '..',
            'KITTI', 'data_road', dataset_type, 'image_2',
            f'{general_name_file}.png'
        )
        if not (os.path.isfile(bin_path) and \
                os.path.isfile(img_path)):
            print(f'Πρόβλημα με το {general_name_file}')
            continue;

        image = cv2.imread(img_path)
        points = load_velodyne_bin(bin_path)

        visualize_toggle(image, points, P2, Tr_velo_to_cam)

    return;


if __name__ == '__main__':
    main()
