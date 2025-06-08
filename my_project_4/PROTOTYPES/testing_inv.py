import glob
import os
import sys
import cv2
import numpy as np
from datetime import datetime

# --- Add CARLA Egg
try:
    carla_egg_path = glob.glob(os.path.abspath(
        os.path.join('..', 'CARLA_0.9.11', 'WindowsNoEditor',
                     'PythonAPI', 'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
                         sys.version_info.major, sys.version_info.minor,
                         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
    ))[0]
    sys.path.append(carla_egg_path)
except:
    print('CARLA egg not found.')
    sys.exit(1)

import carla

# --- Imports from your project
from carla_helpers import get_kitti_calibration, setup_CARLA, setup_camera
from Bii_LiDAR_obstacle_detect import detect_obstacles_withLiDAR

# --- Global buffers
latest_rgb = {'frame': None}
raw_lidar_points = None

def camera_callback(image):
    global latest_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    latest_rgb['frame'] = array

def lidar_callback(point_cloud):
    global raw_lidar_points
    data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
    raw_lidar_points = data[:, :3]

def spawn_invisible_car(world, blueprint_library, location):
    car_bp = blueprint_library.filter('vehicle.audi.a2')[0]
    car = world.spawn_actor(car_bp, carla.Transform(location))
    car.set_simulate_physics(True)
    car.set_autopilot(False)

    # Move it far outside camera frustum (but still in LiDAR range)
    car.set_location(location)
    car.set_light_state(carla.VehicleLightState.NONE)
    return car

def main():
    (world, settings_backup) = setup_CARLA()

    bp_lib = world.get_blueprint_library()
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp_lib.filter('model3')[0], spawn_point)
    vehicle.set_autopilot(True)

    # LiDAR + Camera setup
    WIDTH, HEIGHT, FOV = 600, 600, 90
    camera = setup_camera(world, bp_lib, vehicle, WIDTH, HEIGHT, FOV)
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar = world.spawn_actor(
        lidar_bp, carla.Transform(carla.Location(x=0.2, z=1.8)), attach_to=vehicle
    )

    camera.listen(camera_callback)
    lidar.listen(lidar_callback)

    # Calibration matrices
    _, P2, Tr_velo_to_cam = get_kitti_calibration(
        WIDTH=WIDTH, HEIGHT=HEIGHT, FOV=FOV, camera=camera, lidar=vehicle
    )

    # Spawn invisible test car
    target_car = spawn_invisible_car(
        world, bp_lib,
        location=carla.Location(x=10, y=0, z=0)  # In front of ego vehicle
    )

    print('[INFO] Setup complete.')

    try:
        while True:
            world.tick()

            if latest_rgb['frame'] is None or raw_lidar_points is None:
                continue

            image = latest_rgb['frame'].copy()
            (vis_image, clusters, road_points) = detect_obstacles_withLiDAR(
                image, raw_lidar_points, P2, Tr_velo_to_cam
            )

            cv2.imshow('Test Invisible Object', vis_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print('\nInterrupted')
    finally:
        world.apply_settings(settings_backup)
        for actor in [vehicle, target_car, camera, lidar]:
            actor.destroy()
        cv2.destroyAllWindows()
        print('Cleanup successful.')

if __name__ == '__main__':
    main()
