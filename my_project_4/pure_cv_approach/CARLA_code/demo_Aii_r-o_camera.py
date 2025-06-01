# demo_Aii_r-o_camera.py
# r-o: Road and Obstacles detection using only stereo camera setup!

import os
import sys
import glob
from datetime import datetime

import cv2
import numpy as np

from carla_helpers import (
    get_camera_intrinsic_matrix,
    setup_camera,
    overlay_mask,
    setup_CARLA
)

ai_from_disparity_path = os.path.abspath(
    os.path.join('..', 'Ai_road_finder')
)
sys.path.append(ai_from_disparity_path)
from Ai_from_disparity import (
    detect_ground_mask,
    post_process_mask,
    crop_bottom_half
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

# --- Image buffer
latest_images = {'left': None, 'right': None}

# --- Callbacks
def _cam_callback(buffer_name):
    ''' Constructor wrapper για δημιουργία image callback '''
    def _cb(image):
        array = np.frombuffer(image.raw_data, dtype = np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        latest_images[buffer_name] = array
    
    return _cb;

def main():
    world, original_settings = setup_CARLA()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    # --- Stereo cameras
    (WIDTH, HEIGHT, FOV) = (600, 600, 90)
    # https://www.cvlibs.net/datasets/kitti/setup.php
    camera_2 = setup_camera(
        world, blueprint_library, vehicle,
        WIDTH, HEIGHT, FOV
    )
    baseline = 0.54 # Η απόσταση μεταξύ των 2 καμερών!
    camera_3 = setup_camera(
        world,
        blueprint_library,
        vehicle,
        WIDTH, HEIGHT, FOV,
        y_arg = baseline
    )

    camera_2.listen(_cam_callback('left'))
    camera_3.listen(_cam_callback('right'))
    
    world.tick() # Για να έχουνε 'υπάρξει' στον κόσμο!
    K  = get_camera_intrinsic_matrix(WIDTH, HEIGHT, FOV)
    P2 = np.hstack([K, np.zeros((3, 1))]) # P2 = [K | 0]

    # Compute KITTI-style right camera projection matrix
    # Μετατροπή από το αριστερό στο δεξί σύστημα αναφοράς {camera space}
    T  = np.array([[-baseline], [0], [0]])
    P3 = np.hstack([K, K @ T]) # P3 = K · [I | t]

    f = P2[0, 0]
    calib = {
        'f':  f,
        'cx': P2[0, 2],
        'cy': P2[1, 2],
        'Tx': -(P3[0, 3] - P2[0, 3]) / f
    }

    dt0 = datetime.now()
    try:
        while True:
            world.tick()

            if latest_images['left'] is None or \
                latest_images['right'] is None:
                continue;

            left_color = latest_images['left'].copy()
            left_gray  = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(
                latest_images['right'].copy(), cv2.COLOR_BGR2GRAY
            )

            left_gray  = crop_bottom_half(left_gray)
            right_gray = crop_bottom_half(right_gray)

            mask = detect_ground_mask(
                left_gray,
                right_gray,
                left_color.shape,
                calib,
                ransac_threshold = 0.008,
                crop_bottom = True
            )

            mask = post_process_mask(
                mask, min_area = 2000, kernel_size = 5
            )
            if np.sum(mask) == 0: continue;
            vis = overlay_mask(left_color, mask)

            cv2.imshow('Stereo Aii – Road detection', vis)
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
        camera_2.destroy()
        camera_3.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print('\nΕπιτυχής εκκαθάριση!')

    return;

if __name__ == '__main__':
    main()
