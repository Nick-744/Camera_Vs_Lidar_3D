import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ----------- Configuration -----------
base = os.path.dirname(__file__)
i = 16  # Image index (e.g., um_000017.png)
left_img_path = os.path.join(base, '..', 'data_road', 'training', 'image_2', f'um_0000{i}.png')
right_img_path = os.path.join(base, '..', 'data_road_right', 'training', 'image_3', f'um_0000{i}.png')
calib_path = os.path.join(base, '..', 'data_road', 'training', 'calib', f'um_0000{i}.txt')
# -------------------------------------

def load_calibration(calib_path, matrix_type_1='P2', matrix_type_2='P3'):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    calib_matrix_1 = calib_matrix_2 = None
    for line in lines:
        if line.startswith(matrix_type_1):
            calib_matrix_1 = np.array(line.strip().split()[1:], dtype='float32').reshape(3, -1)
        elif line.startswith(matrix_type_2):
            calib_matrix_2 = np.array(line.strip().split()[1:], dtype='float32').reshape(3, -1)
    return calib_matrix_1, calib_matrix_2

def show_pointcloud(points, colors):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Flip Z axis for correct view (camera facing forward)
    ax.scatter(points[:, 0], -points[:, 1], -points[:, 2],
               c=colors / 255.0, s=0.5, depthshade=True)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (depth)')

    ax.view_init(elev=10, azim=-90)  # Adjust as needed
    plt.tight_layout()
    plt.show()

def main():
    # Load images
    img_left_color = cv2.imread(left_img_path)
    img_right_color = cv2.imread(right_img_path)
    if img_left_color is None or img_right_color is None:
        print("Error: Could not load images.")
        return

    # Convert to grayscale and blur
    img_left_gray = cv2.cvtColor(img_left_color, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right_color, cv2.COLOR_BGR2GRAY)
    img_left_gray = cv2.GaussianBlur(img_left_gray, (5, 5), 0)
    img_right_gray = cv2.GaussianBlur(img_right_gray, (5, 5), 0)

    # StereoSGBM setup
    min_disp = 0
    num_disp = 128  # Must be divisible by 16
    block_size = 5

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute disparity
    disparity = stereo.compute(img_left_gray, img_right_gray).astype(np.float32) / 16.0

    # Load calibration and compute Q matrix
    calib_matrix_1, calib_matrix_2 = load_calibration(calib_path)
    if calib_matrix_1 is None or calib_matrix_2 is None:
        print("Error: Calibration data missing.")
        return

    cam1 = calib_matrix_1[:, :3]
    cam2 = calib_matrix_2[:, :3]
    T = np.array([0.54, 0.0, 0.0])  # KITTI baseline in meters
    Q = np.zeros((4, 4))

    cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2,
                      distCoeffs1=None, distCoeffs2=None,
                      imageSize=img_left_color.shape[:2],
                      R=np.eye(3), T=T,
                      R1=None, R2=None, P1=None, P2=None, Q=Q)

    # Reproject to 3D
    plt.imshow(disparity, cmap='plasma')
    plt.title("Disparity Map")
    plt.colorbar()
    plt.show()

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = (disparity > 1.0) & np.isfinite(points_3D).all(axis=2)

    # Extract valid points and colors
    valid_points = points_3D[mask]
    valid_colors = img_left_color[mask]

    # Optional filtering for visualization clarity
    z = valid_points[:, 2]

    print(f"Total 3D points: {points_3D.shape[0]*points_3D.shape[1]}")
    print(f"Valid points after mask: {np.sum(mask)}")

    # Show 3D result
    show_pointcloud(valid_points, valid_colors)

if __name__ == "__main__":
    main()
