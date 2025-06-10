import numpy as np
import cv2
import airsim

# --- Setups
def setup_AirSim() -> airsim.CarClient:
    """Sets up AirSim car client and returns it."""
    import time
    
    client = airsim.CarClient()
    
    # Try to connect with retries
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to AirSim (attempt {attempt + 1}/{max_retries})...")
            client.confirmConnection()
            print("Successfully connected to AirSim!")
            break
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("Failed to connect to AirSim after all attempts.")
                print("Please make sure:")
                print("1. AirSim is running")
                print("2. You have the correct msgpack version: pip install msgpack==1.0.0")
                raise
    
    try:
        client.enableApiControl(True)
        client.armDisarm(True)
        
        # Reset to a good initial state
        client.reset()
    except Exception as e:
        print(f"Warning: Could not complete full setup: {e}")
        print("Continuing anyway...")
    
    return client

def setup_camera_settings(client: airsim.CarClient,
                         camera_name: str,
                         WIDTH: int,
                         HEIGHT: int,
                         FOV: float) -> None:
    """Configure camera settings in AirSim."""
    # AirSim uses different camera configuration approach
    # Camera settings are typically configured in settings.json
    # This is a placeholder for runtime configuration if needed
    pass

# --- Helpers
def get_camera_intrinsic_matrix(width: int,
                               height: int,
                               fov_deg: float) -> np.ndarray:
    """
    The intrinsic matrix allows you to transform 3D coordinates
    to 2D coordinates on an image plane using the pinhole camera model!
    
    https://ksimek.github.io/2013/08/13/intrinsic/
    """
    fov_rad = np.radians(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx * (height / width)
    (cx, cy) = (width / 2, height / 2)

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

def get_transform_matrix(pose: airsim.Pose) -> np.ndarray:
    """
    airsim.Pose -> Homogeneous Transformation Matrix [world coords]
    """
    # Convert quaternion to rotation matrix
    q = pose.orientation
    x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val
    
    # Quaternion to rotation matrix conversion
    rotation_matrix = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    
    # Position
    position = pose.position
    
    # Build homogeneous transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[0, 3] = position.x_val
    matrix[1, 3] = position.y_val
    matrix[2, 3] = position.z_val
    
    return matrix

def get_kitti_calibration(WIDTH: int,
                         HEIGHT: int,
                         FOV: float,
                         baseline: float = -1.,
                         client: airsim.CarClient = None,
                         camera_name: str = "0",
                         lidar_name: str = "LidarSensor1") -> tuple:
    """
    Calculates internal and external calibration parameters
    for KITTI dataset compatible simulation!

    Params:
     - baseline: Distance between cameras (-1 for LiDAR queries)!
     - client: AirSim client for getting sensor poses
     - camera_name: Camera sensor name
     - lidar_name: LiDAR sensor name

    Returns:
    - calib: Dictionary with basic calibration parameters.
    - P2: Left camera projection matrix.
    - Tr_velo_to_cam: Transformation from LiDAR coordinate system
      to camera coordinate system.
    """
    K = get_camera_intrinsic_matrix(WIDTH, HEIGHT, FOV)
    P2 = np.hstack([K, np.zeros((3, 1))])  # P2 = [K | 0]
    f = P2[0, 0]

    if baseline == -1 and client is not None:
        try:
            # Get camera pose
            camera_info = client.simGetCameraInfo(camera_name)
            camera_transform = get_transform_matrix(camera_info.pose)
            
            # For AirSim, we need to get LiDAR data to access its pose
            # This is a workaround since AirSim doesn't have a direct getLidarPose method
            lidar_data = client.getLidarData(lidar_name)
            lidar_transform = get_transform_matrix(lidar_data.pose)
            
            # Compute relative transformation
            lidar_to_camera = np.linalg.inv(camera_transform) @ lidar_transform
            
            # AirSim coordinate system conversion to KITTI format
            # AirSim: +X forward, +Y right, +Z down (NED)
            # KITTI: +X right, +Y down, +Z forward
            axis_conv = np.array([
                [0, 1, 0, 0],  # X_cam = Y_airsim
                [0, 0, 1, 0],  # Y_cam = Z_airsim  
                [1, 0, 0, 0],  # Z_cam = X_airsim
                [0, 0, 0, 1]
            ])
            
            Tr_velo_to_cam = axis_conv @ lidar_to_camera
            calib = None
            
        except Exception as e:
            print(f"Warning: Could not get sensor poses for calibration: {e}")
            print("Using identity transformation as fallback...")
            # Fallback to identity transformation
            Tr_velo_to_cam = np.eye(4)
            calib = None
    else:
        T = np.array([[-baseline], [0], [0]])
        P3 = np.hstack([K, K @ T])  # P3 = K · [I | t]

        calib = {
            'f': f,
            'cx': P2[0, 2],
            'cy': P2[1, 2],
            'Tx': -(P3[0, 3] - P2[0, 3]) / f
        }
        
        Tr_velo_to_cam = None

    return (calib, P2, Tr_velo_to_cam)

def overlay_mask(image: np.ndarray,
                mask: np.ndarray,
                color: tuple = (0, 0, 255),
                alpha: float = 0.5) -> np.ndarray:
    """Apply transparent mask overlay on image"""
    # Indices of mask pixels that are foreground
    idx = mask.astype(bool)

    # Create solid color for the mask
    solid = np.empty_like(image[idx])
    solid[:] = color

    # Combine original image with solid color (mask)
    image[idx] = cv2.addWeighted(image[idx], 1 - alpha, solid, alpha, 0)

    return image

def airsim_image_to_opencv(airsim_image: airsim.ImageResponse) -> np.ndarray:
    """Convert AirSim image response to OpenCV format."""
    if airsim_image.pixels_as_float:
        # For depth images or other float images
        img_array = np.array(airsim_image.image_data_float, dtype=np.float32)
        img_array = img_array.reshape(airsim_image.height, airsim_image.width)
        img_array = np.expand_dims(img_array, axis=2)
    else:
        # For RGB images
        img_array = np.frombuffer(airsim_image.image_data_uint8, dtype=np.uint8)
        img_array = img_array.reshape(airsim_image.height, airsim_image.width, 3)
        # AirSim returns BGR, convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    return img_array