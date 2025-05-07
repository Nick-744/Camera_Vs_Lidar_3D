#!/usr/bin/env python3
"""
capture_kitti_from_carla.py
===========================

Spawn a KITTI-accurate sensor rig in CARLA (Velodyne HDL-64E + left RGB
camera) and **dump every frame to KITTI-style files**:

out/
├── velodyne/
│   ├── 000000.bin
│   ├── 000001.bin
│   └── …
└── image_2/
    ├── 000000.png
    ├── 000001.png
    └── …

Each `.bin` contains `[x y z reflectance]` as little-endian `float32`,
exactly like the official KITTI raw dataset.  The RGB image is saved lossless
as PNG (1392 × 1024).

Usage
-----
$ python capture_kitti_from_carla.py --frames 300 --out out_dir
Press **Ctrl-C** to stop early.
"""
from __future__ import annotations

import argparse
import glob
import queue
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ─────────────────── CARLA egg ──────────────────────────────────────────────
egg_dir = Path("../../CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist")
egg = glob.glob(str(egg_dir / f"carla-*{sys.version_info.major}.{sys.version_info.minor}-*.egg"))
if not egg:
    print("[ERROR] CARLA egg not found – adjust *egg_dir* inside the script.")
    sys.exit(1)
sys.path.append(egg[0])
import carla  # noqa: E402  pylint: disable=wrong-import-position

# ─────────────────── Local module: KITTI rig ────────────────────────────────
from kitti_setup import KittiSetup, IMG_WIDTH, IMG_HEIGHT


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def carla_to_kitti_velo(pts: np.ndarray) -> np.ndarray:
    """CARLA sensor frame → KITTI Velodyne axes   (x,-y,z)."""
    return np.stack([pts[:, 0], -pts[:, 1], pts[:, 2]], axis=-1)


def ensure_dirs(root: Path) -> Tuple[Path, Path]:
    vel_dir = root / "velodyne"
    img_dir = root / "image_2"
    vel_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    return vel_dir, img_dir


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=300,
                        help="how many frames to dump (0 = unlimited)")
    parser.add_argument("--out", type=str, default="out",
                        help="output root directory")
    args = parser.parse_args()

    vel_dir, img_dir = ensure_dirs(Path(args.out))

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # synchronous mode
    orig_settings = world.get_settings()
    sync = world.get_settings()
    sync.synchronous_mode = True
    sync.fixed_delta_seconds = 0.05
    world.apply_settings(sync)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    actors: list[carla.Actor] = []
    try:
        # ── ego vehicle
        vehicle_bp = world.get_blueprint_library().filter("model3")[0]
        vehicle = world.spawn_actor(
            vehicle_bp,
            random.choice(world.get_map().get_spawn_points())
        )
        vehicle.set_autopilot(True)
        actors.append(vehicle)

        # ── KITTI sensor rig
        kitti = KittiSetup(world, vehicle)
        sensors = kitti.spawn()
        actors.extend(sensors.values())
        lidar = sensors["lidar"]
        cam   = sensors["cam_left"]

        # queues for synchronous capture
        q_lidar: queue.Queue[carla.LidarMeasurement] = queue.Queue()
        q_img:   queue.Queue[carla.Image]            = queue.Queue()
        lidar.listen(q_lidar.put)
        cam.listen(q_img.put)

        frame = 0
        print("[INFO] Recording …  (Ctrl+C to stop)")
        while args.frames == 0 or frame < args.frames:
            world.tick()

            try:
                lidar_meas = q_lidar.get(timeout=2.0)
                img_meas   = q_img.get(timeout=2.0)
            except queue.Empty:
                print("[WARN] sensor timeout – skipped frame")
                continue

            # ── save LiDAR (.bin)
            lidar_data = np.frombuffer(lidar_meas.raw_data, np.float32).reshape(-1, 4)
            pts_kitti  = np.c_[carla_to_kitti_velo(lidar_data[:, :3]),
                               lidar_data[:, 3]]
            (vel_dir / f"{frame:06d}.bin").write_bytes(pts_kitti.tobytes())

            # ── save RGB image (.png)
            rgb = np.frombuffer(img_meas.raw_data, np.uint8) \
                    .reshape(img_meas.height, img_meas.width, 4)[..., :3]
            cv2.imwrite(str(img_dir / f"{frame:06d}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if frame and frame % 50 == 0:
                print(f"[INFO] saved frame {frame:06d}")

            frame += 1

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted – stopping capture.")
    finally:
        print("[INFO] Cleaning up …")
        for a in actors:
            if a.is_alive:
                a.destroy()
        tm.set_synchronous_mode(False)
        world.apply_settings(orig_settings)
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
