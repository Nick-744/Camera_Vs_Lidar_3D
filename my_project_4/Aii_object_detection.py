import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)

temp_left = os.path.join(
    base_dir,
    'data_road',
    'testing',
    'image_2',
    f'um_000000.png'
)
temp_right = os.path.join(
    base_dir,
    'data_road_right',
    'testing',
    'image_3',
    f'um_000000.png'
)
left_img_file = os.path.abspath(temp_left)
right_img_file = os.path.abspath(temp_right)

# Load your rectified stereo pair (replace these with your image paths)
left_img = cv2.imread(left_img_file, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_img_file, cv2.IMREAD_GRAYSCALE)

# Ensure same size
assert left_img.shape == right_img.shape, "Images must be the same size"

# Create StereoSGBM matcher
min_disp = 0
num_disp = 16 * 6  # must be divisible by 16
block_size = 7     # size of matching block (usually 3–11)

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 1 * block_size ** 2,
    P2=32 * 1 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# --- Disparity Map ---
disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

# --- Fix invalid disparity values ---
disparity[disparity <= 0.0] = 0.1

# --- Convert to Depth (meters) ---
focal_length = 721.5377
baseline = 0.54
depth = (focal_length * baseline) / disparity

# --- Display Depth Map ---
depth_vis = np.clip(depth, 0, 50)
plt.imshow(depth_vis, cmap='plasma')
plt.colorbar(label='Depth (m)')
plt.title('Depth Map from Stereo')
plt.axis('off')
plt.show()
