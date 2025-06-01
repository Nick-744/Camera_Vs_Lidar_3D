# demo_Aii_r-o_camera.py
# r-o: Road and Obstacles detection using only stereo camera setup!

"""
This script streams images from a stereo camera rig in CARLA, reconstructs a
3-D point-cloud from their disparity, extracts the ground plane and projects a
road mask back to the left image.  A single **sign change** in the stereo
calibration (`Tx = +fx * baseline`) fixes the negative-depth issue that was
blocking the pipeline.

* 1 × CARLA client (vehicle set to autopilot)
* 1 × RGB left camera, 1 × RGB right camera (baseline ≈ 0.54 m)
* OpenCV SGBM for disparity, RANSAC for ground extraction
"""

import glob
import os
import sys
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 3rd-party / project-local modules
# ---------------------------------------------------------------------------

# 1️⃣  CARLA Python API egg ---------------------------------------------------
try:
    carla_egg = glob.glob(os.path.abspath(
        os.path.join('..', '..', 'CARLA_0.9.11', 'WindowsNoEditor',
                     'PythonAPI', 'carla', 'dist',
                     f"carla-*{sys.version_info.major}.{sys.version_info.minor}-"
                     f"{'win-amd64' if os.name == 'nt' else 'linux-x86_64'}.egg")
    ))[0]
    sys.path.append(carla_egg)
except IndexError:
    print('[ERROR] Could not locate CARLA egg. Check path in script header.')
    sys.exit(1)

import carla  # noqa: E402 (after egg appended)

# 2️⃣  Project helpers --------------------------------------------------------
from carla_helpers import (  # type: ignore
    setup_CARLA,
    setup_camera,
    get_camera_intrinsic_matrix,
    overlay_mask,
)

# Import Aii module
sys.path.append(os.path.abspath(os.path.join('..', 'Ai_road_finder')))
from Ai_from_disparity import (
    compute_disparity, point_cloud_from_disparity,
    ransac_ground, project_points_to_mask,
    post_process_mask
)

# ---------------------------------------------------------------------------
# Globals for latest camera frames
# ---------------------------------------------------------------------------
latest_images = {'left': None, 'right': None}

def _cam_callback(buffer_name):
    """Factory creating a CARLA image callback that stores images in RAM."""
    def _cb(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        latest_images[buffer_name] = array
    return _cb

# ---------------------------------------------------------------------------
# 2-D → 3-D projection helper (with robust filtering)
# ---------------------------------------------------------------------------

def project_points_to_mask(points_3d, calib, img_shape, offset_y=0):
    """Project ground-plane points onto a binary mask in image space."""

    if points_3d is None or len(points_3d) == 0:
        print('[DEBUG] No 3-D points supplied for projection.')
        return None

    pts = points_3d.copy()
    neg_ratio = np.mean(pts[:, 2] < 0)
    if neg_ratio > 0.9:
        print(f'[DEBUG] Flipping Z axis (negZ ratio: {neg_ratio:.2f})')
        pts[:, 2] *= -1

    # --- Coarse physical gates to reject strays --------------------------------
    dist = (pts[:, 2] > 1.0) & (pts[:, 2] < 200.0)
    side = np.abs(pts[:, 0]) < 100.0
    vert = np.abs(pts[:, 1]) < 50.0
    mask = dist & side & vert
    if not np.any(mask):
        # fallback: ultra-lenient
        dist = (pts[:, 2] > 0.5) & (pts[:, 2] < 1000.0)
        side = np.abs(pts[:, 0]) < 500.0
        vert = np.abs(pts[:, 1]) < 500.0
        mask = dist & side & vert
        if not np.any(mask):
            print('[DEBUG] No points survive even lenient filtering.')
            return None
        print('[DEBUG] Using very lenient filtering.')

    pts = pts[mask]
    fx, fy = calib['fx'], calib['fy']
    cx, cy = calib['cx'], calib['cy']

    u = fx * pts[:, 0] / pts[:, 2] + cx
    v = fy * pts[:, 1] / pts[:, 2] + cy + offset_y

    u = u.astype(int)
    v = v.astype(int)

    h, w = img_shape[:2]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_img):
        print('[DEBUG] Projected points fall outside image bounds.')
        return None

    mask_img = np.zeros((h, w), dtype=np.uint8)
    mask_img[v[in_img], u[in_img]] = 255
    mask_img = cv2.dilate(mask_img, np.ones((5, 5), np.uint8), iterations=2)
    return mask_img

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main():
    # 1.  Spawn world & ego-vehicle --------------------------------------------------
    world, original_settings = setup_CARLA()
    bp_lib = world.get_blueprint_library()

    vehicle_bp = bp_lib.filter('model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, world.get_map().get_spawn_points()[0])
    vehicle.set_autopilot(True)

    # 2.  Stereo cameras ------------------------------------------------------------
    W, H, FOV = 600, 600, 90
    cam_left  = setup_camera(world, bp_lib, vehicle, W, H, FOV)
    cam_right = setup_camera(world, bp_lib, vehicle, W, H, FOV, y_arg=0.54)

    cam_left.listen(_cam_callback('left'))
    cam_right.listen(_cam_callback('right'))
    world.tick()  # let sensors spin up

    # 3.  Intrinsics & stereo calibration ------------------------------------------
    K = get_camera_intrinsic_matrix(W, H, FOV)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    B = 0.54  # metres

    crop = H // 3  # ignore sky
    calib = {
        'f': fx, 'fx': fx, 'fy': fy,
        'cx': cx, 'cy': cy - crop,
        'Tx': fx * B,  # +ve! depth = f*Tx/d
        'baseline': B,
    }

    print(f'[INFO] fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, baseline={B:.2f}')
    print(f'[INFO] Using cropped cy = {calib["cy"]:.1f}\n')

    prev_time = datetime.now()
    try:
        while True:
            world.tick()

            if latest_images['left'] is None or latest_images['right'] is None:
                continue

            left_img  = latest_images['left']
            right_img = latest_images['right']

            # --- Pre-processing ---------------------------------------------------
            gray_l = cv2.cvtColor(left_img,  cv2.COLOR_BGR2GRAY)[crop:, :]
            gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)[crop:, :]

            # --- Disparity --------------------------------------------------------
            disp = compute_disparity(gray_l, gray_r)
            good = disp[disp > 0]
            if good.size:
                print(f'[DEBUG] Disparity range: {good.min():.2f} – {good.max():.2f} px | '
                      f'mean {good.mean():.2f}')
                print(f'[DEBUG] Depth (approx): {fx*B/good.max():.2f} – {fx*B/good.min():.2f} m')

            # --- 3-D reconstruction ---------------------------------------------
            pts = point_cloud_from_disparity(disp, calib)
            if pts is None or not len(pts):
                print('[WARN ] No 3-D points from disparity; skipping frame.')
                continue
            print(f'[DEBUG] 3-D points: {len(pts):,}')

            # --- Ground plane -----------------------------------------------------
            _, ground_pts, _ = ransac_ground(pts, distance_threshold=1.0)
            if ground_pts is None or not len(ground_pts):
                print('[WARN ] No ground inliers.')
                continue

            # --- Projection back to image ----------------------------------------
            mask = project_points_to_mask(ground_pts, calib, left_img.shape, offset_y=crop)
            if mask is None or not np.any(mask):
                print('[WARN ] Empty mask after projection.')
                continue

            mask = post_process_mask(mask, min_area=2000, kernel_size=5) or mask
            vis = overlay_mask(left_img.copy(), mask)

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

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
