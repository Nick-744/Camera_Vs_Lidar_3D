#!/usr/bin/env python3
"""Open3D visualiser for a **KITTI‑accurate** sensor rig in CARLA.

This is the original real‑time LiDAR road‑detection demo rewritten to use the
:pyclass:`~kitti_setup.KittiSetup` class so that the LiDAR (HDL‑64E) and the
stereo RGB cameras sit in the exact same relative poses, with the very same
intrinsic parameters, as on the real KITTI recording platform.

Make sure the sibling ``kitti_setup.py`` file is on your PYTHONPATH.
"""
from __future__ import annotations

import glob
import os
import sys
import random
import time
from datetime import datetime
from pathlib import Path
from scipy.spatial import cKDTree 

import numpy as np
import open3d as o3d
import cv2
from matplotlib import cm

# ---------------------------------------------------------------------------
# CARLA egg import (expects $PWD/../carla/dist)
# ---------------------------------------------------------------------------
try:
    carla_egg_pattern = '../../CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-*.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
    )
    sys.path.append(glob.glob(carla_egg_pattern)[0])
except IndexError:
    print('[ERROR] CARLA egg not found. Please set the correct path.')
    sys.exit(1)

import carla  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Local modules – KITTI sensor rig + road detection stub
# ---------------------------------------------------------------------------
from kitti_setup import (
    KittiSetup,
    IMG_WIDTH as WIN_W,
    IMG_HEIGHT as WIN_H,
    FOCAL_LENGTH,
    PRINCIPAL_POINT,
)

try:
    temp = '../part_B'
    sys.path.append(glob.glob(str(Path(__file__).parent / temp))[0])
except (IndexError, ImportError):
    print('[ERROR] Cannot locate Bi_road_detection_pcd.py on PYTHONPATH.')
    sys.exit(1)

from Bi_road_detection_pcd import my_road_from_pcd_is  # type: ignore

# ---------------------------------------------------------------------------
# Colour map constants
# ---------------------------------------------------------------------------
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

# Projection matrix for the *left* KITTI camera (camera 2 in KITTI RAW)
P2 = np.array(
    [
        [FOCAL_LENGTH, 0.0, PRINCIPAL_POINT[0], 0.0],
        [0.0, FOCAL_LENGTH, PRINCIPAL_POINT[1], 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def transform_to_matrix(transform: carla.Transform) -> np.ndarray:
    """Convert CARLA Transform → 4×4 homogeneous matrix (world)."""
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


# ---------------------------------------------------------------------------
# LiDAR callback factory
# ---------------------------------------------------------------------------

def make_lidar_callback(point_list, lidar, camera,
                        Tr_velo_to_cam, P2_left):
    """Return a closure that visualises the LiDAR and overlays road mask."""

    # Once‑off colour LUT for intensity
    def intensity_to_rgb(iarr):
        i_col = 1.0 - np.log(iarr) / np.log(np.exp(-0.004 * 100))
        return np.c_[
            np.interp(i_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(i_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(i_col, VID_RANGE, VIRIDIS[:, 2]),
        ]

    cam_buf: dict[str, np.ndarray | None] = {"rgb": None}

    def cam_cb(img):
        cam_buf["rgb"] = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3]

    camera.listen(cam_cb)

    def lidar_cb(point_cloud):
        # ---- raw to numpy --------------------------------------------------
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
        pts_raw = data[:, :3]
        pts_vis = pts_raw.copy(); pts_vis[:, 1] *= -1  # Y‑flip for Open3D
        colors = intensity_to_rgb(data[:, 3])

        # ---- road detection ----------------------------------------------
        rgb = cam_buf["rgb"]
        if rgb is not None:
            img_h, img_w = rgb.shape[:2]

            # --- LiDAR → CAMERA optical frame ----------------------------------
            pts_cam = (Tr_velo_to_cam @ np.c_[pts_raw, np.ones(len(pts_raw))].T)[:3].T
            pts_cv  = np.empty_like(pts_cam)          # CARLA → OpenCV axis swap
            pts_cv[:, 0] =  pts_cam[:, 1]             #  +X_cam  = +Y_carla
            pts_cv[:, 1] = -pts_cam[:, 2]             #  +Y_cam  = −Z_carla
            pts_cv[:, 2] =  pts_cam[:, 0]             #  +Z_cam  = +X_carla

            # -------------------------------------------------------------------
            road_mask, road_pts, _ = my_road_from_pcd_is(
                pts_cv.astype(np.float32),
                np.eye(4),
                P2_left.astype(np.float32),
                (img_h, img_w),
                debug=False,
                apply_filters=False,
            )

            # -------- colour LiDAR points that fall on the road ---------------
            if road_pts.size:
                tree   = cKDTree(road_pts, compact_nodes=True, balanced_tree=True)
                hit    = tree.query_ball_point(pts_cv, r=0.10)          # 10 cm tol
                onroad = np.fromiter((len(h) > 0 for h in hit),
                                    dtype=bool, count=len(pts_vis))
                colors[onroad] = [0.0, 0.0, 1.0]

            # -------- nice 2‑D overlay ----------------------------------------
            mask_rgb               = np.zeros_like(rgb)
            mask_rgb[road_mask>0]  = [255, 0, 0]
            cv2.imshow("Road mask (LiDAR)", cv2.addWeighted(rgb, 0.6, mask_rgb, 0.4, 0))
            cv2.waitKey(1)

        # ---- push to Open3D ----------------------------------------------
        point_list.points = o3d.utility.Vector3dVector(pts_vis)
        point_list.colors = o3d.utility.Vector3dVector(colors)

    return lidar_cb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.)
    world = client.get_world()

    # ---- synchronous mode -----------------------------------------------
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    actors = []
    vis = None
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('model3')[0]
        spawn_pt = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_pt)
        vehicle.set_autopilot(True)
        actors.append(vehicle)

        # ---- KITTI sensor rig -------------------------------------------
        kitti = KittiSetup(world, vehicle)
        sensors = kitti.spawn()
        lidar = sensors['lidar']
        camera = sensors['cam_left']  # left rectified camera
        actors.extend(sensors.values())

        P2_left          = kitti.P2_left              # 3×4  (float32)
        Tr_velo_to_cam   = kitti.Tr_velo_to_cam       # 4×4  (float32)

        # ---- Open3D initialisation --------------------------------------
        point_list = o3d.geometry.PointCloud()
        vis = o3d.visualization.Visualizer()
        vis.create_window('CARLA HDL‑64E (KITTI)', WIN_W, WIN_H)
        ro = vis.get_render_option()
        ro.background_color = [0.05, 0.05, 0.05]
        ro.point_size = 1.0
        ro.show_coordinate_frame = True

        # ---- register callback & loop -----------------------------------
        lidar.listen(make_lidar_callback(point_list, lidar, camera,
                                         Tr_velo_to_cam, P2_left))
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

            # FPS read‑out --------------------------------------------------
            now = datetime.now()
            fps = 1.0 / (now - t_prev).total_seconds()
            t_prev = now
            frame += 1
            # print(f"\rFPS: {fps:5.2f}", end='')

    except KeyboardInterrupt:
        print('\n[INFO] User interruption – exiting …')
    finally:
        if vis is not None:
            vis.destroy_window()
        cv2.destroyAllWindows()
        for a in actors:
            if a.is_alive:
                a.destroy()
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)


if __name__ == '__main__':
    main()
