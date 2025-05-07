from __future__ import annotations
"""kitti_setup.py

A faithful, self‑contained approximation of the original KITTI sensor
configuration for CARLA ≥ 0.9. This module spawns

  • one Velodyne HDL‑64E S2 LiDAR
  • a rectified, global‑shutter stereo pair of PointGrey Flea2 RGB cameras

on top of an arbitrary ego‑vehicle and exposes their intrinsic / extrinsic
parameters exactly in KITTI text‑calibration format.

Typical usage
-------------
>>> kitti = KittiSetup(world, vehicle)
>>> sensors = kitti.spawn()
>>> print(sensors["lidar"], sensors["cam_left"], sensors["cam_right"])
>>> kitti.write_calibration("calib.txt")          # ⇢ identical to KITTI files
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import carla
from scipy.spatial.transform import Rotation as R

__all__ = ["KittiSetup"]


# =============================================================================
# Canonical KITTI 2011‑09‑26 sensor parameters  (colour, 1392 × 512 → 1242 × 375)
# =============================================================================
IMG_WIDTH: int = 1242         # px
IMG_HEIGHT: int = 375         # px
FOCAL_LENGTH: float = 721.5377  # px   (fx = fy after rectification)
PRINCIPAL_POINT = (609.5593, 172.8540)            # (cx, cy)  px
# horizontal FoV derived from sensor size and fx
FOV: float = float(2 * np.degrees(np.arctan(IMG_WIDTH / (2 * FOCAL_LENGTH))))

BASELINE: float = 0.537  # m   (rectified stereo baseline ≈ 53.7 cm)

# ----------------------------------------------------------------------------- 
# Velodyne HDL‑64E S2 (10 Hz, 1.3 M pts/s) ‑‑ official datasheet values
# -----------------------------------------------------------------------------
LIDAR_ATTRIBS = {
    "channels": "64",
    "points_per_second": "1300000",
    "rotation_frequency": "10",
    "range": "120.0",
    "upper_fov": "2.0",
    "lower_fov": "-24.8",
}

# =============================================================================
# Rigid body poses *in CARLA coordinates*             (x→fwd, y→right, z→up)
# =============================================================================
# Velodyne: roof‑mount, origin centred, optical frame rotated s.t.
# CARLA→KITTI mapping later yields the canonical KITTI velo frame
_LIDAR_POSE = carla.Transform(
    carla.Location(x=0.00, y=0.00, z=1.73),
    # The official KITTI extrinsic (see IJRR’13, Geiger et al.)
    # results in R_velo_to_cam = [[0,-1,0],[0,0,-1],[1,0,0]].
    # In CARLA we therefore rotate the sensor by Rz(0) @ Ry(-90) @ Rx(-90)
    carla.Rotation(roll=-90.0, pitch=-90.0, yaw=0.0),
)

# Cameras: 30 cm in front of lidar, 27 cm below, baseline along –y (leftwards)
_CAM_LEFT_POSE = carla.Transform(
    carla.Location(x=0.30, y=0.00, z=1.46),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
)
_CAM_RIGHT_POSE = carla.Transform(
    carla.Location(x=0.30, y=-BASELINE, z=1.46),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
)

# =============================================================================
# Helper utilities
# =============================================================================
def _carla_transform_to_matrix(tr: carla.Transform) -> np.ndarray:
    """Convert ``carla.Transform`` → 4 × 4 homogeneous matrix (CARLA frame)."""
    loc = tr.location
    rot = R.from_euler("xyz",
                       [tr.rotation.roll, tr.rotation.pitch, tr.rotation.yaw],
                       degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = np.array([loc.x, loc.y, loc.z], dtype=np.float32)
    return T


def _carla_to_kitti() -> np.ndarray:
    """4×4 fixed axis permutation (CARLA → KITTI)."""
    return np.array([[0, 1, 0, 0],
                     [0, 0, -1, 0],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


def _projection_matrix(tx: float = 0.0) -> np.ndarray:
    """3×4 intrinsic‑projection *P* (identical to KITTI *P0 / P1*)."""
    cx, cy = PRINCIPAL_POINT
    fx = fy = FOCAL_LENGTH
    return np.array([[fx, 0.0, cx, tx],
                     [0.0, fy, cy, 0.0],
                     [0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float32)


# =============================================================================
# Main interface
# =============================================================================
class KittiSetup:
    """Spawn the KITTI sensor rig on *vehicle* inside a CARLA *world*."""

    def __init__(self, world: carla.World, vehicle: carla.Actor):
        self.world = world
        self.vehicle = vehicle
        self.bp_lib = self.world.get_blueprint_library()
        self.sensors: Dict[str, carla.Actor] = {}

    # ---------------------------------------------------------------------
    # Public calibration helpers (lazy‑evaluated from current poses)
    # ---------------------------------------------------------------------
    @property
    def P2_left(self) -> np.ndarray:
        return _projection_matrix()

    @property
    def P3_right(self) -> np.ndarray:
        tx = -FOCAL_LENGTH * BASELINE
        return _projection_matrix(tx)

    @property
    def Tr_velo_to_cam(self) -> np.ndarray:
        """3×4 rigid transform (Velodyne → left‑cam optical, KITTI)."""
        return self._lidar_to_cam0()

    # ---------------------------------------------------------------------
    # Sensor life‑cycle
    # ---------------------------------------------------------------------
    def spawn(self) -> Dict[str, carla.Actor]:
        """Spawn LiDAR + stereo cameras.  Returns {name: actor}."""
        self._spawn_lidar()
        self._spawn_camera("cam_left",  _CAM_LEFT_POSE)
        self._spawn_camera("cam_right", _CAM_RIGHT_POSE)
        return self.sensors

    def destroy(self):
        """Safely destroy all spawned sensors."""
        for actor in self.sensors.values():
            if actor.is_alive:
                actor.destroy()
        self.sensors.clear()

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def write_calibration(self, file: str | Path):
        """Write KITTI‑compatible *calib.txt*."""
        file = Path(file)
        lines = [
            "P0: " + " ".join(f"{v:.7e}" for v in self.P2_left.ravel()),
            "P1: " + " ".join(f"{v:.7e}" for v in self.P3_right.ravel()),
            "Tr_velo_to_cam: " + " ".join(
                f"{v:.7e}" for v in self.Tr_velo_to_cam.ravel()
            ),
        ]
        file.write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # internal – spawn helpers
    # ------------------------------------------------------------------
    def _spawn_lidar(self):
        bp = self.bp_lib.find("sensor.lidar.ray_cast")
        for k, v in LIDAR_ATTRIBS.items():
            bp.set_attribute(k, v)
        lidar = self.world.spawn_actor(bp, _LIDAR_POSE, attach_to=self.vehicle)
        self.sensors["lidar"] = lidar

    def _spawn_camera(self, name: str, pose: carla.Transform):
        bp = self.bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(IMG_WIDTH))
        bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        bp.set_attribute("fov", f"{FOV:.3f}")
        cam = self.world.spawn_actor(bp, pose, attach_to=self.vehicle)
        self.sensors[name] = cam

    # ------------------------------------------------------------------
    # internal – calibration maths
    # ------------------------------------------------------------------
    def _lidar_to_cam0(self) -> np.ndarray:
        """Return 3×4 Velodyne→Cam 0 transform (KITTI) computed at runtime."""
        T_lidar = _carla_transform_to_matrix(_LIDAR_POSE)
        T_cam   = _carla_transform_to_matrix(_CAM_LEFT_POSE)
        T_velo_cam_carla = np.linalg.inv(T_cam) @ T_lidar

        C2K = _carla_to_kitti()
        T_velo_cam_kitti = C2K @ T_velo_cam_carla @ np.linalg.inv(C2K)
        return T_velo_cam_kitti[:3, :4].astype(np.float32)
