# demo_Bi_road_detection_airsim.py

import os
import sys
import cv2
import numpy as np
import time
import airsim
from datetime import datetime

# --- Imports ---
from airsim_helpers import (
    get_kitti_calibration,
    overlay_mask,
    setup_AirSim,
    airsim_image_to_opencv
)

# Add path for road detection function from pcd
part_B_module_path = os.path.abspath(os.path.join('..', 'part_B'))
sys.path.append(part_B_module_path)
from Bi_road_detection_pcd import my_road_from_pcd_is

def main():
    # Setup AirSim connection
    client = setup_AirSim()
    print('AirSim setup completed!')

    # Configuration
    WIDTH, HEIGHT, FOV = 600, 600, 90
    CAMERA_NAME = "0"  # Default front camera
    LIDAR_NAME = "LidarSensor1"
    
    # Enable car controls for movement (optional)
    car_controls = airsim.CarControls()
    car_controls.throttle = 0.3
    car_controls.steering = 0.0
    client.setCarControls(car_controls)

    # Get calibration matrices
    (_, P2, Tr_velo_to_cam) = get_kitti_calibration(
        WIDTH=WIDTH, HEIGHT=HEIGHT, FOV=FOV,
        client=client, camera_name=CAMERA_NAME, lidar_name=LIDAR_NAME
    )

    print('Starting main loop. Press "q" to quit.')

    # Main loop
    dt0 = datetime.now()
    try:
        while True:
            # Get camera image
            image_request = airsim.ImageRequest(
                CAMERA_NAME, airsim.ImageType.Scene, False, False
            )
            image_response = client.simGetImages([image_request])[0]
            
            if image_response is None:
                print("No image received, skipping frame...")
                time.sleep(0.05)
                continue

            # Convert AirSim image to OpenCV format
            rgb_image = airsim_image_to_opencv(image_response)
            
            # Resize image to match our configuration
            if rgb_image.shape[:2] != (HEIGHT, WIDTH):
                rgb_image = cv2.resize(rgb_image, (WIDTH, HEIGHT))

            # Get LiDAR data
            lidar_data = client.getLidarData(LIDAR_NAME)
            
            if len(lidar_data.point_cloud) < 3:
                print("No LiDAR points received, skipping frame...")
                time.sleep(0.05)
                continue

            # Convert LiDAR data to numpy array
            raw_lidar_points = np.array(
                lidar_data.point_cloud, dtype=np.float32
            ).reshape(-1, 3)

            if raw_lidar_points.shape[0] == 0 or np.isnan(raw_lidar_points).any():
                print("Invalid point cloud, skipping frame...")
                time.sleep(0.05)
                continue

            # Process road detection
            display = rgb_image.copy()
            try:
                (mask, _, _) = my_road_from_pcd_is(
                    raw_lidar_points,
                    Tr_velo_to_cam,
                    P2,
                    (HEIGHT, WIDTH)
                )

                # Apply mask overlay
                display = overlay_mask(
                    display, mask, color=(255, 0, 0), alpha=0.5
                )
            except Exception as e:
                print(f"Road detection error: {e}")
                # Continue with original image if road detection fails

            # Display result
            cv2.imshow('AirSim Dash Camera - Road Detection', display)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Calculate and display FPS
            dt1 = datetime.now()
            fps = 1.0 / max((dt1 - dt0).total_seconds(), 0.001)
            print(f'\rFPS: {fps:.2f}', end='', flush=True)
            dt0 = dt1

            # Small delay to prevent overwhelming the system
            time.sleep(0.01)

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nError occurred: {e}')
    finally:
        # Cleanup
        print('\nCleaning up...')
        
        # Stop the car
        car_controls.throttle = 0.0
        car_controls.brake = 1.0
        client.setCarControls(car_controls)
        
        # Reset AirSim
        client.reset()
        client.enableApiControl(False)
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        print('Cleanup completed successfully!')

    return

if __name__ == '__main__':
    main()