# demo_Aii_r-o_camera.py
# r-o: Road and Obstacles detection using only stereo camera setup!

import glob
import os
import sys
from datetime import datetime

import cv2
import numpy as np

from carla_helpers import (  # type: ignore
    setup_CARLA,
    setup_camera,
    get_camera_intrinsic_matrix,
    overlay_mask
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

# Import Aii module
ai_from_disparity_path = os.path.abspath(
    os.path.join('..', 'Ai_road_finder')
)
sys.path.append(ai_from_disparity_path)
from Ai_from_disparity import (detect_ground_mask, post_process_mask)

# --- Image buffer
latest_images = {'left': None, 'right': None}

# --- Callbacks
def _cam_callback(buffer_name):
    """Factory creating a CARLA image callback that stores images in RAM."""
    def _cb(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        latest_images[buffer_name] = array
    return _cb

def main():
    # 1.  Spawn world & ego-vehicle --------------------------------------------------
    world, original_settings = setup_CARLA()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    # 2.  Stereo cameras ------------------------------------------------------------
    (WIDTH, HEIGHT, FOV) = (600, 600, 90)
    cam_left  = setup_camera(
        world, blueprint_library, vehicle, WIDTH, HEIGHT, FOV
    )
    cam_right = setup_camera(
        world, blueprint_library, vehicle, WIDTH, HEIGHT, FOV, y_arg = 0.54
    )

    cam_left.listen(_cam_callback('left'))
    cam_right.listen(_cam_callback('right'))
    world.tick()  # let sensors spin up

    # 3.  Intrinsics & stereo calibration ------------------------------------------
    K = get_camera_intrinsic_matrix(WIDTH, HEIGHT, FOV)
    P2 = np.hstack([K, np.zeros((3, 1))]) # P2 = [K | 0]

    # Compute KITTI-style right camera projection matrix
    baseline = 0.54 # Πρέπει να είναι το ίδιο με το y_arg του cam_right!
    # Μετατροπή από το αριστερό στο δεξί σύστημα αναφοράς {camera space}
    T = np.array([[-baseline], [0], [0]])
    P3 = np.hstack([K, K @ T]) # P3 = K · [I | t]

    f  = P2[0, 0]
    cx = P2[0, 2]
    cy = P2[1, 2]
    Tx = -(P3[0, 3] - P2[0, 3]) / f
    calib = {'f': f, 'cx': cx, 'cy': cy, 'Tx': Tx}

    prev_time = datetime.now()
    try:
        while True:
            world.tick()

            if latest_images['left'] is None or latest_images['right'] is None:
                continue

            left_color = latest_images['left'].copy()
            left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(latest_images['right'].copy(), cv2.COLOR_BGR2GRAY)

            mask = detect_ground_mask(
                left_gray,
                right_gray,
                left_color.shape,
                calib,
                ransac_threshold = 6,
                #crop_bottom = True
            )

            mask = post_process_mask(mask, min_area=2000, kernel_size=5)
            vis = overlay_mask(left_color, mask)

            cv2.imshow('Stereo Aii – Road detection', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # --- FPS --------------------------------------------------------------
            now = datetime.now()
            fps = 1.0 / max((now - prev_time).total_seconds(), 1e-3)
            prev_time = now
            print(f'FPS: {fps:5.1f}', end='\r')

    finally:
        # -----------------------------------------------------------------------
        world.apply_settings(original_settings)
        cam_left.destroy()
        cam_right.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print('\n[INFO] Shutdown complete.')


if __name__ == '__main__':
    main()
