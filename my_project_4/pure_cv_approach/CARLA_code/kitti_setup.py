from __future__ import annotations
"""kitti_setup.py

A minimal, self‑contained KITTI sensor rig description for CARLA 0.9.x.

Instantiate ``KittiSetup`` with a ``carla.World`` and a reference vehicle to
spawn the sensors (HDL‑64E LiDAR + stereo RGB cameras) at the same relative
poses, resolutions and intrinsic parameters as the original KITTI dataset
recorded by the Autonomous Driving group at KIT.

Example
-------
>>> kitti = KittiSetup(world, vehicle)
>>> sensors = kitti.spawn()
>>> print(sensors["lidar"], sensors["cam_left"], sensors["cam_right"])

The class also exposes the intrinsic and extrinsic calibration matrices in the
exact KITTI text format via :py:meth:`write_calibration` for easy downstream
use.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import carla
from scipy.spatial.transform import Rotation as R

__all__ = [
    "KITTI_ROOT",
    "KittiSetup",
]

# ---------------------------------------------------------------------------
# Configuration constants (taken from the official KITTI sensor reference)
# ---------------------------------------------------------------------------

KITTI_ROOT = Path("KITTI_Dataset_CARLA")

# Image size and intrinsics for the rectified colour cameras (PNG 12‑bit → 8‑bit)
IMG_WIDTH = 1392
IMG_HEIGHT = 1024
FOCAL_LENGTH = 721.5377  # pixels (≈ 35 mm focal length on this sensor size)
PRINCIPAL_POINT = (609.5593, 172.854)
FOV = 72.0  # horizontal field of view (deg) – used when setting Carla camera fov

# Baseline between the rectified stereo pair (in metres)
BASELINE = 0.537  # KITTI average ≈ 0.537 m

# LiDAR (Velodyne HDL‑64E S2) parameters
LIDAR_ATTRIBS = {
    "channels": "64",
    "points_per_second": "1300000",
    "range": "100.0",
    "upper_fov": "2.0",
    "lower_fov": "-24.9",
}

# Relative poses of the sensors w.r.t the vehicle (ego) frame
# Coordinate system conventions:
#   – CARLA:  x → forward,   y → right,  z → up
#   – KITTI:  x → right,     y → down,   z → forward
# To reproduce KITTI, we keep CARLA default, then convert when writing calib.
_LIDAR_POSE = carla.Transform(
    carla.Location(x=0.0, y=0.0, z=1.73),  # roof‑mounted above cameras
    carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),  # spin axis to +z (KITTI fwd)
)
# Cameras sit 30 cm in front of lidar and 17 cm below, baseline along –y
_CAM_LEFT_POSE = carla.Transform(
    carla.Location(x=0.30, y=0.0, z=1.56), carla.Rotation())
_CAM_RIGHT_POSE = carla.Transform(
    carla.Location(x=0.30, y=-BASELINE, z=1.56), carla.Rotation())

# ---------------------------------------------------------------------------
# KITTI ‑ style rigid transform   LiDAR → LEFT RGB camera  (optical frame)
# ---------------------------------------------------------------------------
# Rotation  (KITTI official: x→z, y→−x, z→−y)
R_velo_to_cam = np.array([[ 0., -1.,  0.],
                          [ 0.,  0., -1.],
                          [ 1.,  0.,  0.]], dtype=np.float32)

# Translation (metres) – change these three numbers only
t_velo_to_cam = np.array([0.27, 0.00, 0.00], dtype=np.float32)   # +27 cm X‑forward

# Full 4 × 4 homogeneous matrix
TR_VELO_TO_CAM = np.eye(4, dtype=np.float32)
TR_VELO_TO_CAM[:3, :3] = R_velo_to_cam
TR_VELO_TO_CAM[:3,  3] = t_velo_to_cam

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _kitti_projection_matrix(tx: float = 0.0) -> np.ndarray:
    """Return a 3×4 projection matrix *P* identical to KITTI calib "P0/P1"."""
    cx, cy = PRINCIPAL_POINT
    fx = fy = FOCAL_LENGTH
    P = np.array([[fx, 0.0, cx, tx], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    return P


def _carla_transform_to_matrix(tr: carla.Transform) -> np.ndarray:
    """Convert CARLA transform to 4×4 homogeneous matrix (CARLA frame)."""
    loc = tr.location
    rot = R.from_euler("xyz", [tr.rotation.roll, tr.rotation.pitch, tr.rotation.yaw], degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = np.array([loc.x, loc.y, loc.z])
    return T


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KittiSetup:
    """Spawns the KITTI sensor rig on a *vehicle* inside a CARLA *world*."""

    def __init__(self, world: carla.World, vehicle: carla.Actor, output: Path | str = KITTI_ROOT):
        self.world = world
        self.vehicle = vehicle
        self.output = Path(output)
        self.bp_library = self.world.get_blueprint_library()
        self.sensors: Dict[str, carla.Actor] = {}

    # ------------------------------------------------------------------
    # Public calibration helpers
    # ------------------------------------------------------------------
    @property
    def P2_left(self) -> np.ndarray:
        """3×4 intrinsic‑projection matrix for the left RGB camera."""
        return np.array([[FOCAL_LENGTH, 0.0, PRINCIPAL_POINT[0],  0.0],
                        [0.0,          FOCAL_LENGTH, PRINCIPAL_POINT[1], 0.0],
                        [0.0,          0.0,          1.0,                0.0]],
                        dtype=np.float32)

    @property
    def Tr_velo_to_cam(self) -> np.ndarray:
        """4×4 rigid transform (LiDAR → left camera optical frame)."""
        return TR_VELO_TO_CAM.astype(np.float32)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def spawn(self) -> Dict[str, carla.Actor]:
        """Spawn LiDAR + stereo RGB cameras and return a dict mapping names to actors."""
        self.output.mkdir(parents=True, exist_ok=True)
        self._spawn_lidar()
        self._spawn_camera("cam_left", _CAM_LEFT_POSE)
        self._spawn_camera("cam_right", _CAM_RIGHT_POSE)
        return self.sensors

    def destroy(self):
        """Destroy all spawned sensors."""
        for actor in self.sensors.values():
            if actor.is_alive:
                actor.destroy()
        self.sensors.clear()

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def write_calibration(self, frame_id: int | str = "000000") -> None:
        """Write a KITTI‑formatted *calib* txt into ``output/calib``."""
        calib_dir = self.output / "calib"
        calib_dir.mkdir(parents=True, exist_ok=True)

        # Rectified camera projection matrices (baseline encoded as ±tx)
        tx_left = 0.0
        tx_right = -FOCAL_LENGTH * BASELINE
        P0 = _kitti_projection_matrix(tx_left)
        P1 = _kitti_projection_matrix(tx_right)

        # Identity rectification matrix (R0_rect) as in KITTI raw → odometry
        R0_rect = np.eye(3, dtype=np.float32)

        # LiDAR → cam0 rigid transform (Tr_velo_to_cam) in KITTI coords
        Tr_velo_to_cam = self._lidar_to_cam0()

        def _fmt(mat: np.ndarray) -> str:
            return " ".join(map(lambda x: f"{x:.6f}", mat.flatten()))

        with open(calib_dir / f"{str(frame_id).zfill(6)}.txt", "w") as fh:
            fh.write(f"P0: {_fmt(P0)}\n")
            fh.write(f"P1: {_fmt(P1)}\n")
            fh.write(f"R0_rect: {_fmt(R0_rect)}\n")
            fh.write(f"Tr_velo_to_cam: {_fmt(Tr_velo_to_cam)}\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_lidar(self):
        lidar_bp = self.bp_library.find("sensor.lidar.ray_cast")
        for k, v in LIDAR_ATTRIBS.items():
            lidar_bp.set_attribute(k, v)
        lidar = self.world.spawn_actor(lidar_bp, _LIDAR_POSE, attach_to=self.vehicle)
        self.sensors["lidar"] = lidar

    def _spawn_camera(self, name: str, pose: carla.Transform):
        cam_bp = self.bp_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        cam_bp.set_attribute("fov", str(FOV))
        camera = self.world.spawn_actor(cam_bp, pose, attach_to=self.vehicle)
        self.sensors[name] = camera

    # ------------------------------------------------------------------
    # Calibration math
    # ------------------------------------------------------------------

    def _lidar_to_cam0(self) -> np.ndarray:
        """Return 3×4 rigid transform from LiDAR to left camera (KITTI frame)."""
        T_lidar = _carla_transform_to_matrix(_LIDAR_POSE)
        T_cam0 = _carla_transform_to_matrix(_CAM_LEFT_POSE)

        # LiDAR → ego and cam0 → ego, so T_cam0_ego = ...; We need velo → cam
        T_ego_to_cam0 = np.linalg.inv(T_cam0)
        T_velo_to_cam0 = T_ego_to_cam0 @ T_lidar

        # Convert CARLA (x fwd, y right, z up) → KITTI (x right, y down, z fwd)
        C = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        Tr = C @ T_velo_to_cam0 @ np.linalg.inv(C)
        return Tr[:3]  # 3×4
