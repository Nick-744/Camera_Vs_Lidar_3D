import glob
import os
import sys
import cv2
import numpy as np
from datetime import datetime

# Προσθήκη του path για την συνάρτηση εύρεσης του δρόμου από το pcd!
bi_road_module_path = os.path.abspath(os.path.join('..', 'part_B'))
sys.path.append(bi_road_module_path)
from Bi_road_detection_pcd import my_road_from_pcd_is

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
latest_rgb = {'frame': None}
raw_lidar_points = None

# --- Helpers
def get_camera_intrinsic_matrix(width:   int,
                                height:  int,
                                fov_deg: float) -> np.ndarray:
    '''
    The intrinsic matrix allows you to transform 3D coordinates
    to 2D coordinates on an image plane using the pinhole camera model!
    
    https://ksimek.github.io/2013/08/13/intrinsic/
    '''
    fov_rad = np.radians(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx * (height / width)
    (cx, cy) = (width / 2, height / 2)

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ]);

def get_transform_matrix(transform: carla.Transform) -> np.ndarray:
    '''
    carla.Transform -> Homogeneous Transformation Matrix [world coords]
    '''
    rotation = transform.rotation
    location = transform.location
    (cy, sy) = (
        np.cos(np.radians(rotation.yaw)), np.sin(np.radians(rotation.yaw))
    )
    (cp, sp) = (
        np.cos(np.radians(rotation.pitch)), np.sin(np.radians(rotation.pitch))
    )
    (cr, sr) = (
        np.cos(np.radians(rotation.roll)), np.sin(np.radians(rotation.roll))
    )

    matrix = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location.x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location.y],
        [sp,     -cp * sr,                                cp * cr, location.z],
        [0,             0,                                      0,          1]
    ])

    return matrix;

def overlay_mask(image: np.ndarray,
                 mask:  np.ndarray,
                 color: tuple = (0, 0, 255),
                 alpha: float = 0.5) -> np.ndarray:
    ''' Προβολή διαφανούς μάσκας σε εικόνα '''
    # Οι δείκτες των pixels μάσκας που είναι foreground
    idx = mask.astype(bool)

    # Δημιουργία solid χρώματος για την μάσκα
    solid = np.empty_like(image[idx])
    solid[:] = color

    # Συνδυασμός της αρχικής εικόνας με το solid χρώμα (mask)
    image[idx] = cv2.addWeighted(image[idx], 1 - alpha, solid, alpha, 0)

    return image;

# --- Callbacks
def lidar_callback(point_cloud: carla.LidarMeasurement) -> None:
    global raw_lidar_points

    data = np.frombuffer(
        point_cloud.raw_data, dtype = np.float32
    ).reshape(-1, 4)
    raw_lidar_points = data[:, :3]

    return;

def camera_callback(image: carla.Image) -> None:
    array = np.frombuffer(image.raw_data, dtype = np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    latest_rgb['frame'] = array

    return;

# --- Setups
def setup_CARLA() -> tuple:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.)

    world = client.get_world()
    original_settings = world.get_settings()

    # Για να δουλέψει το vehicle.set_autopilot(True) στην main:
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = True
    
    world.apply_settings(settings)

    return (world, original_settings);

def setup_camera(world,
                 blueprint_library,
                 vehicle,
                 WIDTH:  int,
                 HEIGHT: int,
                 FOV:    float) -> carla.Sensor:
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(WIDTH))
    camera_bp.set_attribute('image_size_y', str(HEIGHT))
    camera_bp.set_attribute('fov', str(FOV))
    camera_transform = carla.Transform(carla.Location(x = 1.5, z = 1.8))
    camera = world.spawn_actor(
        camera_bp, camera_transform, attach_to = vehicle
    )
    
    return camera;

def setup_lidar_sensor(world,
                       blueprint_library,
                       vehicle) -> carla.Sensor:
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '1000000')
    lidar_transform = carla.Transform(carla.Location(x = 0.2, z = 1.8))
    lidar = world.spawn_actor(
        lidar_bp, lidar_transform, attach_to = vehicle
    )
    
    return lidar;

def main():
    (world, original_settings) = setup_CARLA()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    (WIDTH, HEIGHT, FOV) = (600, 600, 90)
    camera = setup_camera(
        world,
        blueprint_library,
        vehicle,
        WIDTH,
        HEIGHT,
        FOV
    )
    lidar = setup_lidar_sensor(world, blueprint_library, vehicle)

    camera.listen(camera_callback)
    lidar.listen(lidar_callback)

    world.tick() # Για να φορτώσουν σωστά οι τιμές:
    K = get_camera_intrinsic_matrix(WIDTH, HEIGHT, FOV)
    P2 = np.hstack([K, np.zeros((3, 1))]) # P2 = [K | 0]

    # Tr_velo_to_cam : LiDAR → Camera (4×4) [με KITTI axes]
    lidar_to_camera = np.linalg.inv(
        get_transform_matrix(camera.get_transform())
    ) @ get_transform_matrix(lidar.get_transform())

    # KITTI έχει: +Z forward, +X right, +Y down (OpenCV).  
    # CARLA/LiDAR frame είναι +X forward, +Y right, +Z up!
    axis_conv = np.array(
        [[0, 1, 0, 0], # X_cam =  Y_lidar
         [0, 0,-1, 0], # Y_cam = -Z_lidar
         [1, 0, 0, 0], # Z_cam =  X_lidar
         [0, 0, 0, 1]]
    )

    Tr_velo_to_cam = axis_conv @ lidar_to_camera

    print('Το setup ολοκληρώθηκε!')

    # --- Main loop
    frame = 0
    dt0 = datetime.now()
    try:
        while True:
            world.tick()

            if (latest_rgb['frame'] is None) or \
                (raw_lidar_points is None):
                continue;

            display = latest_rgb['frame'].copy()
            (mask, _, _) = my_road_from_pcd_is(
                raw_lidar_points,
                Tr_velo_to_cam,
                P2,
                (HEIGHT, WIDTH)
            )

            # Εφαρμογή μάσκας στο display
            display = overlay_mask(
                display, mask, color = (255, 0, 0)
            )
            cv2.imshow('', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;

            # FPS
            dt1 = datetime.now()
            fps = 1. / (dt1 - dt0).total_seconds()
            print(f'\rFPS: {fps:.2f}', end = '')
            dt0 = dt1
            
            frame += 1

    except KeyboardInterrupt:
        print('\nΔιακοπή')
    finally:
        world.apply_settings(original_settings)
        lidar.destroy()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print('\nΕπιτυχής εκκαθάριση!')

    return;

if __name__ == '__main__':
    main()
