# demo_Aii_r-o_camera.py
# r-o: Road and Obstacles detection using only stereo camera setup!

import glob
import os
import sys
import numpy as np
import cv2
from datetime import datetime

# Import Aii module
sys.path.append(os.path.abspath(os.path.join('..', 'Ai_road_finder')))
from Ai_from_disparity import (
    compute_disparity, point_cloud_from_disparity,
    ransac_ground, project_points_to_mask,
    overlay_mask, post_process_mask
)

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

latest_images = {'left': None, 'right': None}

# --- Camera callback

def left_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    latest_images['left'] = array

def right_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    latest_images['right'] = array

# --- Intrinsics calculator

def get_intrinsics(width, height, fov_deg):
    fov_rad = np.radians(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    cx, cy = width / 2, height / 2

    return fx, cx, cy

# --- Main setup and loop

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.)

    world = client.get_world()
    original_settings = world.get_settings()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    WIDTH, HEIGHT, FOV = 600, 600, 90

    # Stereo cameras
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(WIDTH))
    camera_bp.set_attribute('image_size_y', str(HEIGHT))
    camera_bp.set_attribute('fov', str(FOV))

    left_cam = world.spawn_actor(
        camera_bp, carla.Transform(carla.Location(x=1.5, y=-0.15, z=1.8)), attach_to=vehicle)
    right_cam = world.spawn_actor(
        camera_bp, carla.Transform(carla.Location(x=1.5, y=0.15, z=1.8)), attach_to=vehicle)

    left_cam.listen(left_callback)
    right_cam.listen(right_callback)

    fx, cx, cy = get_intrinsics(WIDTH, HEIGHT, FOV)
    B = 0.3  # Stereo baseline in meters (based on 0.15 + 0.15)

    calib = {'f': fx, 'cx': cx, 'cy': cy, 'Tx': -fx * B}

    print('Stereo Aii setup complete.')

    dt0 = datetime.now()
    try:
        while True:
            world.tick()
            if latest_images['left'] is None or latest_images['right'] is None:
                continue

            left = latest_images['left']
            right = latest_images['right']

            gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            gray_l = gray_l[HEIGHT//2:, :]
            gray_r = gray_r[HEIGHT//2:, :]

            disparity = compute_disparity(gray_l, gray_r)
            points = point_cloud_from_disparity(disparity, calib)
            (_, ground_pts, _) = ransac_ground(points, distance_threshold = 0.5, show = True)
            mask =  project_points_to_mask(
                ground_pts, calib, left.shape, crop_bottom = True
            )

            if mask is None:
                print("[WARNING] mask is None, skipping frame.")
                continue;
            
            cv2.imshow("Raw ground mask", mask)
            cv2.waitKey(1)
            
            # min_area = 15000 <-> 20000 καλύτερα αποτελέσματα!
            road_mask_cleaned = post_process_mask(
                mask, min_area = 0, kernel_size = 7
            )
            if road_mask_cleaned is None or np.count_nonzero(road_mask_cleaned) == 0:
                print("[WARNING] Cleaned mask is empty, skipping frame.")
                continue;

            vis = overlay_mask(left.copy(), road_mask_cleaned)

            cv2.imshow('Stereo Aii Obstacle Detection', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;

            dt1 = datetime.now()
            fps = 1. / (dt1 - dt0).total_seconds()
            print(f'FPS: {fps:.2f}', end='\r')
            dt0 = dt1

    finally:
        world.apply_settings(original_settings)
        left_cam.destroy()
        right_cam.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print('\nCleaned up.')

if __name__ == '__main__':
    main()
