#!/usr/bin/env python
"""Open3D LiDAR visualisation for CARLA **with real‑time LiDAR‑based road detection**.

This script is a self‑contained version of the original CVC example. All command‑line
argument handling has been removed – just run it, and it will:
  • start a synchronous CARLA client on localhost:2000
  • spawn a Tesla Model 3 with an RGB camera and a LiDAR
  • feed the LiDAR point cloud to your existing   `my_road_from_pcd_is()` function
    (imported from `lidar_road_detection.py` – make sure it is on your PYTHONPATH)
  • colour road points blue in the Open3D viewer and overlay a red road mask on
    the RGB camera image in a pop‑up OpenCV window
"""

import glob
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2
from matplotlib import cm

# -----------------------------------------------------------------------------
# CARLA egg import (expects $PWD/../carla/dist)
# -----------------------------------------------------------------------------
try:
    carla_egg_pattern = '../../CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-*.egg' % (
        sys.version_info.major,
        sys.version_info.minor
    )
    sys.path.append(glob.glob(carla_egg_pattern)[0])
except IndexError:
    print('[ERROR] CARLA egg not found. Please set the correct path.')
    sys.exit(1)

import carla  # noqa: E402  pylint: disable=wrong-import-position

# -----------------------------------------------------------------------------
# Your road‑detection module ----------------------------------------------------
# -----------------------------------------------------------------------------
# Provide  my_road_from_pcd_is(pcd, Tr_velo_to_cam, image_shape, apply_filters=True)
try:
    # Πρέπει να είναι στο PYTHONPATH! Το directory που περιέχει
    # το Bi_road_detection_pcd.py, όχι το ίδιο το αρχείο!
    temp = '../part_B'
    sys.path.append(
        glob.glob(str(Path(__file__).parent / temp))[0]
    )
except ImportError as exc:  # pragma: no cover
    print("[ERROR] Cannot import lidar_road_detection.py –", exc)
    sys.exit(1)

from Bi_road_detection_pcd import my_road_from_pcd_is  # type: ignore

# -----------------------------------------------------------------------------
# Colour maps & constants ------------------------------------------------------
# -----------------------------------------------------------------------------
VIRIDIS = np.array(cm.get_cmap("plasma").colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

# Open3D viewer window size
WIN_W, WIN_H = 960, 540
# Camera intrinsics (match WIN_W / WIN_H and FOV)
CAM_FOV = 90. # degrees
FX = FY = (WIN_W / 2.) / np.tan(np.deg2rad(CAM_FOV / 2.))
CX, CY = WIN_W / 2., WIN_H / 2.

# -----------------------------------------------------------------------------
# Utility ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def transform_to_matrix(transform: carla.Transform) -> np.ndarray:
    """Convert CARLA Transform to 4×4 homogeneous matrix (worldframe)."""
    loc = transform.location
    rot = transform.rotation

    cy, sy = np.cos(np.deg2rad(rot.yaw)), np.sin(np.deg2rad(rot.yaw))
    cp, sp = np.cos(np.deg2rad(rot.pitch)), np.sin(np.deg2rad(rot.pitch))
    cr, sr = np.cos(np.deg2rad(rot.roll)), np.sin(np.deg2rad(rot.roll))

    R = np.array(
        [
            [cp * cy, cy * sp * sr - cr * sy, -cr * cy * sp - sr * sy],
            [cp * sy, cr * cy + sp * sr * sy, -cy * sr + cr * sp * sy],
            [sp, -cp * sr, cr * cp],
        ]
    )
    T = np.array([loc.x, loc.y, loc.z])

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = T
    return M


# -----------------------------------------------------------------------------
# Main callback ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def make_lidar_callback(point_list, lidar, camera, P2):
    """Factory to capture external references inside the closure."""

    # Intensity → RGB lookup table (once)
    def intensity_to_rgb(intensity_arr: np.ndarray) -> np.ndarray:
        i_col = 1.0 - np.log(intensity_arr) / np.log(np.exp(-0.004 * 100))
        return np.c_[
            np.interp(i_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(i_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(i_col, VID_RANGE, VIRIDIS[:, 2]),
        ]

    camera_buffer = {"rgb": None}

    def camera_cb(img):
        camera_buffer["rgb"] = (
            np.frombuffer(img.raw_data, dtype=np.uint8).
            reshape((img.height, img.width, 4))[:, :, :3]
        )

    camera.listen(camera_cb)

    def lidar_cb(point_cloud):
        # ---------------- Raw LiDAR → numpy ----------------------------------
        data = np.copy(
            np.frombuffer(
                point_cloud.raw_data,
                dtype = np.float32
            ).reshape(-1, 4)
        )
        pts_raw = data[:, :3]
        pts_vis = pts_raw.copy()
        pts_vis[:, 1] *= -1. # flip Y -> right‑handed (ΜΟΝΟ για Open3D)!

        colors = intensity_to_rgb(data[:, 3])

        # ---------------- Road detection (needs an RGB frame) ----------------
        rgb = camera_buffer["rgb"]
        if rgb is not None:
            img_h, img_w = rgb.shape[:2]

            # Extrinsics:   LiDAR → World → Camera
            Tr_lidar_to_world = transform_to_matrix(lidar.get_transform())
            Tr_cam_to_world = transform_to_matrix(camera.get_transform())
            Tr_world_to_cam = np.linalg.inv(Tr_cam_to_world)
            Tr_lidar_to_cam = Tr_world_to_cam @ Tr_lidar_to_world

            # 2. LiDAR → Camera (CARLA)
            pts_cam = (Tr_lidar_to_cam @ np.c_[pts_raw, np.ones(len(pts_raw))].T)[:3].T

            # 3. Camera (CARLA) → OpenCV/KITTI frame  ➜  [x_cv, y_cv, z_cv]
            pts_cv = np.empty_like(pts_cam)
            pts_cv[:, 0] =  pts_cam[:, 1]        # X_cv = +Y_carla   (δεξιά)
            pts_cv[:, 1] = -pts_cam[:, 2]        # Y_cv = -Z_carla   (κάτω)
            pts_cv[:, 2] =  pts_cam[:, 0]        # Z_cv = +X_carla   (μπροστά)

            try:
                road_mask, road_pts, _ = my_road_from_pcd_is(
                    pts_cv,
                    np.eye(4), # Tr_velo_to_cam = Ι, γιατί τα σημεία είναι ήδη στο cam frame
                    P2, (img_h, img_w),
                    debug = False,
                    apply_filters=False
                )

                # --- paint road points blue ---------------------------------
                road_selector = (np.linalg.norm(pts_vis[:, None] - road_pts[None], axis=-1) < 1e-6).any(axis=1)
                colors[road_selector] = [0.0, 0.0, 1.0]

                # --- RGB overlay -------------------------------------------
                mask_color = np.zeros_like(rgb)
                mask_color[road_mask == 255] = [255, 0, 0]
                overlay = cv2.addWeighted(rgb, 0.6, mask_color, 0.4, 0)
                cv2.imshow("Road mask (LiDAR)", overlay)
                cv2.waitKey(1)
            except Exception as exc:  # pragma: no cover – robust against fails
                print("[road‑det]", exc)

        # ---------------- Push to Open3D -------------------------------------
        point_list.points = o3d.utility.Vector3dVector(pts_vis)
        point_list.colors = o3d.utility.Vector3dVector(colors)

    return lidar_cb


# -----------------------------------------------------------------------------
# Main -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    # --- connection ----------------------------------------------------------
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.)
    world = client.get_world()

    # --- synchronous settings ------------------------------------------------
    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    actors = []  # for clean‑up
    vis = None
    try:
        # --- spawn ego vehicle ----------------------------------------------
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("model3")[0]
        spawn_pt = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_pt)
        vehicle.set_autopilot(True)
        actors.append(vehicle)

        # --- camera ---------------------------------------------------------
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(WIN_W))
        cam_bp.set_attribute("image_size_y", str(WIN_H))
        cam_bp.set_attribute("fov", str(CAM_FOV))
        cam_tf = carla.Transform(carla.Location(x=0., z=1.8))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        actors.append(camera)

        focal = (WIN_W / 2) / np.tan(np.deg2rad(CAM_FOV / 2))
        P2 = np.array([[focal, 0, WIN_W / 2, 0],
                    [0, focal, WIN_H / 2, 0],
                    [0,     0,           1, 0]])

        # --- lidar ----------------------------------------------------------
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("points_per_second", "130000") # ~Velodyne
        lidar_bp.set_attribute("upper_fov", "2.0")
        lidar_bp.set_attribute("lower_fov", "-24.9")
        lidar_bp.set_attribute("range", "100.0")
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / settings.fixed_delta_seconds))
        lidar_tf = carla.Transform(carla.Location(x=0., z=1.8))
        lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        actors.append(lidar)

        # --- Open3D visualiser ---------------------------------------------
        point_list = o3d.geometry.PointCloud()
        vis = o3d.visualization.Visualizer()
        vis.create_window("CARLA LiDAR", WIN_W, WIN_H)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # --- register callback ---------------------------------------------
        lidar.listen(make_lidar_callback(point_list, lidar, camera, P2))

        # --- main loop ------------------------------------------------------
        frame = 0
        t_prev = datetime.now()
        while True:
            if frame == 2:
                vis.add_geometry(point_list)
            vis.update_geometry(point_list)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.005)
            world.tick()

            now = datetime.now()
            fps = 1.0 / (now - t_prev).total_seconds()
            t_prev = now
            # sys.stdout.write(f"\rFPS: {fps:5.2f}")
            # sys.stdout.flush()
            frame += 1

    except KeyboardInterrupt:
        print("\n[INFO] User interruption – exiting …")
    finally:
        if vis is not None:
            vis.destroy_window()
        cv2.destroyAllWindows()
        for act in actors:
            act.destroy()
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
