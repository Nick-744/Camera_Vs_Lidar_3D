# demo_Biii_road_detection.py

import os
import sys
import cv2
import glob
import numpy as np
import open3d as o3d
from datetime import datetime

# --- Imports ---
from carla_helpers import (
    get_kitti_calibration,
    setup_camera,
    setup_CARLA
)

# Προσθήκη path για την υλοποίηση των ερωτημάτων του part B!
part_B_module_path = os.path.abspath(os.path.join('..'))
sys.path.append(part_B_module_path)
from Biii_LiDAR_arrowMove import prepare_processed_pcd

# --- CARLA egg setup
try:
    carla_egg_path = glob.glob(os.path.abspath(
        os.path.join('..', '..',
            'CARLA_0.9.11', 'WindowsNoEditor',
            'PythonAPI', 'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
                sys.version_info.major, sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64'
            )
        )
    ))[0]
    sys.path.append(carla_egg_path)
except:
    print('Πρόβλημα εύρεσης του αρχείου CARLA egg.')
    sys.exit(1);
import carla

# --- Image buffer
latest_rgb = {'frame': None}
raw_lidar_points = None

# --- Callbacks
def lidar_callback(point_cloud: carla.LidarMeasurement) -> None:
    global raw_lidar_points

    data = np.frombuffer(
        point_cloud.raw_data, dtype = np.float32
    ).reshape(-1, 4)
    raw_lidar_points = data[:, :3]

    return;

def camera_callback(image: carla.Image) -> None:
    array = np.frombuffer(image.raw_data, dtype = np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    latest_rgb['frame'] = array

    return;

# --- Setups
def setup_lidar_sensor(
    world:             carla.World,
    blueprint_library: carla.BlueprintLibrary,
    vehicle:           carla.Vehicle
) -> carla.Sensor:
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '1000000')
    lidar_transform = carla.Transform(carla.Location(x = 0.2, z = 1.8))
    lidar = world.spawn_actor(
        lidar_bp, lidar_transform, attach_to = vehicle
    )
    
    return lidar;

def main():
    (world, original_settings) = setup_CARLA()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp  = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle     = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    (WIDTH, HEIGHT, FOV) = (600, 600, 90)
    camera = setup_camera(
        world,
        blueprint_library,
        vehicle,
        WIDTH, HEIGHT, FOV
    )
    lidar = setup_lidar_sensor(world, blueprint_library, vehicle)

    camera.listen(camera_callback)
    lidar.listen(lidar_callback)

    (_, P2, Tr_velo_to_cam) = get_kitti_calibration(
        WIDTH = WIDTH, HEIGHT = HEIGHT, FOV = FOV,
        camera = camera, lidar = vehicle
    )

    # --- Διαμόρφωση της προβολής του pcd ---
    theta = np.radians(90)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    Rz    = np.array([ # Rotation matrix
        [cos_t, -sin_t, 0, 0],
        [sin_t,  cos_t, 0, 0],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ])
    flip = np.array([ # Transformation matrix
        [1,  0,  0, 0],
        [0, -1,  0, 0], # Flip Y-axis
        [0,  0,  1, 0],
        [0,  0,  0, 1]
    ])
    T = Rz @ flip

    print('Το setup ολοκληρώθηκε!')

    # --- Main loop
    dt0 = datetime.now()
    try:
        pcd = o3d.geometry.PointCloud() # Empty pcd
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name = 'LiDAR Viewer - Part B',
            width = 800, height = 600
        )
        geometry_added = False
        
        while True:
            world.tick()

            if (latest_rgb['frame'] is None) or \
                (raw_lidar_points is None):
                continue;

            # .copy() -> Assignment destination is read-only
            display = latest_rgb['frame'].copy()

            # Ενημέρωση του pcd με τα νέα δεδομένα LiDAR
            new_pcd = prepare_processed_pcd(
                display,
                raw_lidar_points,
                P2, Tr_velo_to_cam,
                max_length = 6.,
                origin     = np.array([3., 0., 0.]),
            )
            new_pcd.transform(T)
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors

            # Αρχικοποίηση της προβολής του LiDAR
            if not geometry_added:
                vis.add_geometry(pcd)
                geometry_added = True
            else:
                vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            cv2.imshow('Dash Camera', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;

            # FPS
            dt1 = datetime.now()
            fps = 1. / (dt1 - dt0).total_seconds()
            print(f'\rFPS: {fps:.2f}', end = '')
            dt0 = dt1

    except KeyboardInterrupt:
        print('\nΔιακοπή')
    finally:
        world.apply_settings(original_settings)
        lidar.destroy()
        camera.destroy()
        vehicle.destroy()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print('\nΕπιτυχής εκκαθάριση!')

    return;

if __name__ == '__main__':
    main()
