import os
import cv2
import numpy as np
import open3d as o3d
from time import time
from typing import Tuple, List

from part_B.Bii_LiDAR_obstacle_detect import (
    detect_obstacles_withLiDAR,
    project_clusters_with_dots
)
from part_B.Bi_road_detection_pcd import (
    load_velodyne_bin,
    load_calibration,
    project_lidar_to_image
)

def cast_arrow_along_camera(
    points:          np.ndarray,
    obstacles:       List[np.ndarray],
    Tr_velo_to_cam:  np.ndarray,
    max_length:      float = 25., # Δες prepare_processed_pcd
    step:            float = 0.5,
    road_color:      Tuple[float, float, float] = (1, 0, 1),
    collision_color: Tuple[float, float, float] = (1, 0, 0),
    origin:          np.ndarray = np.array([0., 0., 0.]) # Δες prepare_processed_pcd
) -> o3d.geometry.TriangleMesh:
    '''
    Πρόβαλε 1 3D βέλος από το LiDAR scanner origin προς την κατεύθυνση που
    κοιτά η κάμερα! Αν το βέλος συναντήσει εμπόδιο, τότε σταμάτα εκεί και
    αλλάζει το χρώμα του σε κόκκινο [default: collision_color], αλλιώς
    το βέλος φτάνει μέχρι το τέλος του δρόμου ή max_length. Έχει οριστεί
    κατάλληλη μετατόπιση για την Z θέση του.
    '''
    # Κατεύθυνση βάσει του orientation της κάμερας
    R = Tr_velo_to_cam[:3, :3]
    camera_forward = np.array([0., 0., 1.])

    direction = R.T @ camera_forward
    direction = direction / np.linalg.norm(direction)

    # Προβολή των σημείων του δρόμου προς την κατεύθυνση της κάμερας,
    # έτσι ώστε να βρεθεί το μήκος του δρόμου, οπότε και του βέλους!
    projections = points @ direction
    projections = projections[projections > 0]
    road_len    = float(np.max(projections)) \
                  if projections.size > 0 else 5.
    if max_length is not None:
        road_len = min(road_len, max_length)

    # Έλεγχος για εμπόδια κατά μήκος του βέλους
    arrow_len = road_len
    for d in np.arange(step, road_len, step):
        probe = origin + direction * d
        for obs in obstacles:
            if np.any(np.linalg.norm(obs - probe, axis = 1) < 1.):
                arrow_len  = d - step
                road_color = collision_color
                break;
        else:
            continue;
        break;

    # Δημιουργία του βέλους [mesh]
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius = 0.1,
        cone_radius     = 0.2,
        cylinder_height = arrow_len,
        cone_height     = 0.4
    )
    arrow.paint_uniform_color(road_color)

    # Προσαρμογή του βέλους στην κατεύθυνση της κάμερας
    default_axis = np.array([0., 0., 1.])
    if not np.allclose(direction, default_axis):
        axis  = np.cross(default_axis, direction)
        angle = np.arccos(np.clip(
            np.dot(default_axis, direction), -1., 1.
        ))
        if np.linalg.norm(axis) > 1e-6:
            axis     = axis / np.linalg.norm(axis)
            rvec     = axis * angle
            (rot, _) = cv2.Rodrigues(rvec)
            arrow.rotate(rot, center = (0, 0, 0))

    # Προσαρμογή της θέσης του βέλους
    avg_z = np.mean(points[:, 2]) if len(points) > 0 else 0.
    origin[2] = avg_z + 0.15 # Λίγο πιο πάνω από τον δρόμο
    arrow.translate(origin)

    # Δειγματοληψία, γιατί θέλουμε pcd!
    return arrow.sample_points_uniformly(4000);

def prepare_processed_pcd(
    image:          np.ndarray,
    points:         np.ndarray,
    P2:             np.ndarray,
    Tr_velo_to_cam: np.ndarray,
    max_length:     float = 25.,
    origin:         np.ndarray = np.array([0., 0., 0.])
) -> o3d.geometry.PointCloud:
    '''
    Parameters:
     - origin_: Η θέση του LiDAR scanner στο σύστημα συντεταγμένων.
                Προστέθηκε για να διορθωθεί ένα πρόβλημα στο CARLA demo:
                Όταν η αρχή του βέλους είναι στο (0, 0, 0), τότε αυτό
                χτυπά στην μύτη του ego vehicle, οπότε είναι πάντα κόκκινο!
    '''
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

    # --- Βέλος προσανατολισμού ---
    arrow_mesh = cast_arrow_along_camera(
        road_points,
        list(clusters.values()),
        Tr_velo_to_cam,
        max_length = max_length,
        origin     = origin
    )

    return road_pcd + obs_pcd + arrow_mesh;

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
                     Tr_velo_to_cam: np.ndarray,
                     BEV:            bool = False) -> None:
    '''
    Συνδιασμός των Bi και Bii για εύρεση του δρόμου και
    ανίχνευση εμποδίων με LiDAR + προσθήκη βέλους κατεύθυνσης!

    Parameters:
     - BEV_: Αν True, τότε η κάμερα θα είναι σε
             Bird's Eye View (BEV) αρχικά!
    '''
    start = time()
    temp  = (image, points, P2, Tr_velo_to_cam)
    raw_rgb_pcd   = project_colors_on_pcd(*temp)
    processed_pcd = prepare_processed_pcd(*temp)
    print(f'Χρόνος εκτέλεσης: {time() - start:.2f} sec')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width = 1000, height = 600)
    vis.add_geometry(processed_pcd)

    # --- Camera Car/Bird Viewpoint Setup ---
    if not BEV:
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
    
    print('Πάτα space για εναλλαγή προβολής [raw/processed]!\n')
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

        image  = cv2.imread(img_path)
        points = load_velodyne_bin(bin_path)

        visualize_toggle(
            image, points, P2, Tr_velo_to_cam,
            BEV = False
        )

    return;

if __name__ == '__main__':
    main()
