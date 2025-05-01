import os
import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

# ============== RGB PIPELINE ==============
def pegtop_soft_light(img):
    inv = 255 - img
    a = img.astype(np.float32) / 255.0
    b = inv.astype(np.float32) / 255.0
    out = (1 - 2 * b) * a**2 + 2 * b * a
    return np.clip(out * 255, 0, 255).astype(np.uint8)

def preprocess_rgb(img):
    soft = pegtop_soft_light(img)
    hsv = cv2.cvtColor(soft, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
    hsv = hsv.astype(np.uint8)
    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blurred = cv2.medianBlur(img2, 5)
    eroded = cv2.erode(blurred, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return eroded

def apply_roi_mask(img, ratio=0.5):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    h = img.shape[0]
    mask[int(h * ratio):, :] = 1
    return mask

def extract_superpixel_features(img, segments):
    features, positions = [], []
    for label in np.unique(segments):
        mask = (segments == label)
        coords = np.column_stack(np.where(mask))
        if coords.shape[0] == 0: continue
        center = coords.mean(axis=0)
        color = img[mask].mean(axis=0)
        features.append(np.concatenate((center[::-1], color[::-1])))
        positions.append(center[::-1])
    return np.array(features), np.array(positions)

def detect_rgb_obstacles(img):
    roi_mask = apply_roi_mask(img)
    proc = preprocess_rgb(img)
    segments = slic(proc, n_segments=400, compactness=10, start_label=1)
    segments = segments * roi_mask
    features, positions = extract_superpixel_features(proc, segments)
    if len(features) == 0: return []
    labels = DBSCAN(eps=30, min_samples=3).fit(features).labels_
    boxes = []
    for label in np.unique(labels):
        if label == -1: continue
        pts = positions[labels == label]
        if pts.shape[0] < 2: continue
        x1, y1 = np.min(pts, axis=0)
        x2, y2 = np.max(pts, axis=0)
        w, h = x2 - x1, y2 - y1
        if w*h > 200:  # Filter small boxes
            boxes.append((x1, y1, x2, y2))
    return boxes

# ============== STEREO PIPELINE ==============
def compute_disparity(left, right):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=96, blockSize=5,
        P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    return cv2.medianBlur(disp, 5)

def disparity_to_depth(disp, fx, baseline):
    return (fx * baseline) / (disp + 1e-6)

def depth_to_cloud(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - K[0,2]) * z / K[0,0]
    y = (v - K[1,2]) * z / K[1,1]
    cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid = ~np.isnan(cloud).any(axis=1) & (z.flatten() > 0.5) & (z.flatten() < 40)
    return cloud[valid]

def remove_ground(points):
    ransac = RANSACRegressor(residual_threshold=0.1)
    ransac.fit(points[:, :2], points[:, 2])
    pred_z = ransac.predict(points[:, :2])
    return points[points[:, 2] > pred_z + 0.1]

def detect_stereo_obstacles(left, right, K, baseline):
    disp = compute_disparity(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
    depth = disparity_to_depth(disp, K[0,0], baseline)
    points = depth_to_cloud(depth, K)
    if len(points) == 0: return []
    points = remove_ground(points)
    labels = DBSCAN(eps=0.6, min_samples=10).fit(points).labels_
    boxes = []
    for l in np.unique(labels):
        if l == -1: continue
        pts = points[labels == l]
        proj = pts[:, :2]
        x1, y1 = np.min(proj, axis=0)
        x2, y2 = np.max(proj, axis=0)
        w, h = x2 - x1, y2 - y1
        if w * h > 200:  # Filter small clusters
            boxes.append((x1, y1, x2, y2))
    return boxes

# ============== FUSION ==============
def fuse(rgb_boxes, stereo_boxes, dist_thresh=40):
    fused, used = [], set()
    for rb in rgb_boxes:
        rcx = (rb[0]+rb[2])/2
        rcy = (rb[1]+rb[3])/2
        best, bdist = None, float("inf")
        for i, sb in enumerate(stereo_boxes):
            scx = (sb[0]+sb[2])/2
            scy = (sb[1]+sb[3])/2
            d = np.hypot(rcx - scx, rcy - scy)
            if d < dist_thresh and d < bdist:
                best, bdist = i, d
        if best is not None:
            used.add(best)
            sb = stereo_boxes[best]
            fused.append((min(rb[0], sb[0]), min(rb[1], sb[1]), max(rb[2], sb[2]), max(rb[3], sb[3])))
        else:
            fused.append(rb)
    fused += [b for i, b in enumerate(stereo_boxes) if i not in used]
    return fused

# ============== MAIN & VISUALIZATION ==============
def draw_boxes(img, boxes):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imshow("Fused Obstacle Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    base = os.path.dirname(__file__)
    i = 17
    lpath = os.path.join(base, '..', 'data_road', 'testing', 'image_2', f'um_0000{i}.png')
    rpath = os.path.join(base, '..', 'data_road_right', 'testing', 'image_3', f'um_0000{i}.png')
    left, right = cv2.imread(lpath), cv2.imread(rpath)
    if left is None or right is None:
        raise FileNotFoundError("Missing stereo images")

    # KITTI calibration
    K = np.array([[721.5, 0, 609.5], [0, 721.5, 172.8], [0, 0, 1]])
    baseline = 0.54

    rgb_boxes = detect_rgb_obstacles(left)
    stereo_boxes = detect_stereo_obstacles(left, right, K, baseline)
    fused = fuse(rgb_boxes, stereo_boxes)
    draw_boxes(left, fused)

if __name__ == "__main__":
    main()
