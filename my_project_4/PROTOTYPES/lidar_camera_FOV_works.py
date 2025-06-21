import glob
import os
import sys
import time
import cv2
import numpy as np
import open3d as o3d
from matplotlib import cm
from datetime import datetime

# Setup Carla Python API path
sys.path.append(
    glob.glob(
        '../CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major, sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        ))[0]
)
import carla

# --------------------- Global Colormap ---------------------
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

# Shared image buffer
latest_rgb = {"frame": None}

def get_camera_intrinsic_matrix(width, height, fov_deg):
    fov_rad = np.radians(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx * (height / width)
    cx = width / 2
    cy = height / 2
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

def get_transform_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    cy, sy = np.cos(np.radians(rotation.yaw)), np.sin(np.radians(rotation.yaw))
    cp, sp = np.cos(np.radians(rotation.pitch)), np.sin(np.radians(rotation.pitch))
    cr, sr = np.cos(np.radians(rotation.roll)), np.sin(np.radians(rotation.roll))

    matrix = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location.x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location.y],
        [sp,     -cp * sr,                cp * cr,                 location.z],
        [0, 0, 0, 1]
    ])
    return matrix

def transform_lidar_to_camera_convention(points_lidar):
    points_cam = np.zeros_like(points_lidar)
    points_cam[:, 0] = points_lidar[:, 1]  # X_cam = Y_lidar
    points_cam[:, 1] = -points_lidar[:, 2] # Y_cam = -Z_lidar
    points_cam[:, 2] = points_lidar[:, 0]  # Z_cam = X_lidar
    return points_cam

def filter_visible_lidar_points(points_lidar, lidar_to_camera, camera_intrinsic, image_shape):
    points_hom = np.hstack([points_lidar, np.ones((points_lidar.shape[0], 1))])
    points_cam = (lidar_to_camera @ points_hom.T).T

    points_cam_fixed = transform_lidar_to_camera_convention(points_cam[:, :3])

    in_front = points_cam_fixed[:, 2] > 0.1
    points_cam_front = points_cam_fixed[in_front]
    points_lidar_front = points_lidar[in_front]

    uv_hom = (camera_intrinsic @ points_cam_front.T).T
    uv = uv_hom[:, :2] / uv_hom[:, 2][:, np.newaxis]

    h, w = image_shape[:2]
    in_fov = (
        (uv[:, 0] >= 0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )

    # For visualization, return camera-aligned points
    points_for_vis = points_cam_front[in_fov]
    return points_for_vis

def lidar_callback(point_cloud, point_list):
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.float32))
    data = np.reshape(data, (-1, 4))
    points = data[:, :-1]
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    visible_points = filter_visible_lidar_points(
        points, lidar_to_camera_matrix, camera_intrinsic_matrix, (600, 800)
    )
    
    # --- Apply flipping transform ---
    flip_matrix = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    points_flipped = visible_points @ flip_matrix.T

    point_list.points = o3d.utility.Vector3dVector(points_flipped)
    point_list.colors = o3d.utility.Vector3dVector(int_color[:len(visible_points)])

def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    latest_rgb["frame"] = array

# --- Connect to CARLA ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
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
vehicle.set_autopilot(True, traffic_manager.get_port())

lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '100.0')
lidar_bp.set_attribute('rotation_frequency', '20')
lidar_bp.set_attribute('channels', '64')
lidar_bp.set_attribute('points_per_second', '1000000')
lidar_bp.set_attribute('upper_fov', '15.0')
lidar_bp.set_attribute('lower_fov', '-25.0')

lidar_transform = carla.Transform(carla.Location(x=2.0, z=1.8))
camera_transform = carla.Transform(carla.Location(x=1.5, z=1.8))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Wait one tick to update transforms
world.tick()
lidar_to_camera_matrix = np.linalg.inv(get_transform_matrix(camera.get_transform())) @ get_transform_matrix(lidar.get_transform())
camera_intrinsic_matrix = get_camera_intrinsic_matrix(800, 600, 90)

# Debug
print("[Debug] Lidar to Camera matrix:\n", lidar_to_camera_matrix)
print("[Debug] Camera Intrinsic matrix:\n", camera_intrinsic_matrix)

point_list = o3d.geometry.PointCloud()
lidar.listen(lambda data: lidar_callback(data, point_list))
camera.listen(lambda data: camera_callback(data))

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Carla LiDAR', width=960, height=540)
vis.get_render_option().background_color = [0.05, 0.05, 0.05]
vis.get_render_option().point_size = 1

frame = 0
dt0 = datetime.now()
try:
    while True:
        world.tick()
        if latest_rgb["frame"] is not None:
            cv2.imshow('Camera', latest_rgb["frame"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame == 2:
            vis.add_geometry(point_list)
        vis.update_geometry(point_list)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.005)
        process_time = datetime.now() - dt0
        print(f'FPS: {1.0 / process_time.total_seconds():.2f}', end='\r')
        dt0 = datetime.now()
        frame += 1

except Exception as e:
    print("\n[Error]", e)

finally:
    print("\nCleaning up...")
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(False)
    lidar.destroy()
    camera.destroy()
    vehicle.destroy()
    vis.destroy_window()
    cv2.destroyAllWindows()
    print("Done.")
