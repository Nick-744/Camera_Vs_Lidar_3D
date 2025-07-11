import numpy as np
import glob
import cv2
import sys
import os

# CARLA egg setup | Δεν θέλω να φαίνονται υπογραμμίσεις στο αρχείο...
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

# --- Setups
def setup_CARLA(render: bool = False) -> tuple:
    ''' Ρυμίζει το CARLA world κατάλληλα και τον
        επιστρέφει, μαζί με τις default ρυθμίσεις του. '''
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.)

    # Διάλεξε την πίστα στην οποία θα οδηγούμε!
    # ['/Game/Carla/Maps/Town01', '/Game/Carla/Maps/Town01_Opt',
    #  '/Game/Carla/Maps/Town02', '/Game/Carla/Maps/Town02_Opt',
    #  '/Game/Carla/Maps/Town03', '/Game/Carla/Maps/Town03_Opt',
    #  '/Game/Carla/Maps/Town04', '/Game/Carla/Maps/Town04_Opt',
    #  '/Game/Carla/Maps/Town05', '/Game/Carla/Maps/Town05_Opt']
    # client.load_world('Town04_Opt')

    world = client.get_world()
    original_settings = world.get_settings()

    # Για να δουλέψει το vehicle.set_autopilot(True) στην main:
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    if not render:
        settings.no_rendering_mode = True
    
    world.apply_settings(settings)

    return (world, original_settings);

def setup_camera(world:             carla.World,
                 blueprint_library: carla.BlueprintLibrary,
                 vehicle:           carla.Vehicle,
                 WIDTH:             int,
                 HEIGHT:            int,
                 FOV:               float,
                 y_arg:             float = 0.) -> carla.Sensor:
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(WIDTH))
    camera_bp.set_attribute('image_size_y', str(HEIGHT))
    camera_bp.set_attribute('fov', str(FOV))
    camera_transform = carla.Transform(
        carla.Location(x = 1.5, y = y_arg, z = 1.8)
    )
    camera = world.spawn_actor(
        camera_bp, camera_transform, attach_to = vehicle
    )
    
    return camera;

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
        np.cos(np.radians(rotation.yaw)),
        np.sin(np.radians(rotation.yaw))
    )
    (cp, sp) = (
        np.cos(np.radians(rotation.pitch)),
        np.sin(np.radians(rotation.pitch))
    )
    (cr, sr) = (
        np.cos(np.radians(rotation.roll)),
        np.sin(np.radians(rotation.roll))
    )

    matrix = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location.x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location.y],
        [sp,     -cp * sr,                                cp * cr, location.z],
        [0,             0,                                      0,          1]
    ])

    return matrix;

def get_kitti_calibration(WIDTH:    int,
                          HEIGHT:   int,
                          FOV:      float,
                          baseline: float = -1., # Δεν έχει σημασία για τα LiDAR!
                          camera:   carla.Sensor = None,
                          lidar:    carla.Sensor = None) -> tuple:
    '''
    Υπολογίζει τις παραμέτρους εσωτερικού και εξωτερικού calibration
    για προσομοίωση συμβατή με τη μορφή του KITTI dataset!

    Params:
     - baseline_: Η απόσταση μεταξύ των καμερών (-1 για LiDAR ερωτήματα)!
     - camera_:   None για τα stereo/camera ερωτήματα!
     - lidar_:    None για τα stereo/camera ερωτήματα!

    Returns:
    - calib:          Λεξικό με βασικές παραμέτρους βαθμονόμησης.
    - P2:             Τη μήτρα προβολής της αριστερής κάμερας.
    - Tr_velo_to_cam: Τη μετατροπή από το σύστημα συντεταγμένων του
      LiDAR στο σύστημα της κάμερας.

    Σημειώσεις:
    -----------
    - Η εσωτερική βαθμονόμηση (K) υπολογίζεται από τις διαστάσεις της
      εικόνας και το FOV.
    - Το σύστημα συντεταγμένων του KITTI έχει διαφορετικό προσανατολισμό
      από του CARLA, οπότε εφαρμόζεται axis conversion! ***
    - Η μήτρα P3 χρησιμοποιείται εσωτερικά για τον υπολογισμό του Tx.
    '''
    K  = get_camera_intrinsic_matrix(WIDTH, HEIGHT, FOV)
    P2 = np.hstack([K, np.zeros((3, 1))]) # P2 = [K | 0]
    f  = P2[0, 0]

    # Compute KITTI-style right camera projection matrix
    # Μετατροπή από το αριστερό στο δεξί σύστημα αναφοράς {camera space}
    if baseline == -1:
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

        calib = None;
    else:
        T  = np.array([[-baseline], [0], [0]])
        P3 = np.hstack([K, K @ T]) # P3 = K · [I | t]

        calib = {
            'f':  f,
            'cx': P2[0, 2],
            'cy': P2[1, 2],
            'Tx': -(P3[0, 3] - P2[0, 3]) / f
        }

        Tr_velo_to_cam = None;

    return (calib, P2, Tr_velo_to_cam);

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
