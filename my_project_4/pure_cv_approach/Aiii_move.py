from pathlib import Path
from time import time

import cv2
import numpy as np

from Ai_road_finder.Ai_from_disparity import parse_kitti_calib
from Aii_object_detection.Aii_obj_detection_current import (
    detect_obstacles,
    crop_bottom_half,
    overlay_mask,
    draw_bboxes,
)

# colours (BGR)
GREEN = (0, 255, 0)
RED   = (0, 0, 255)
ROAD  = (255, 0, 0)  # blue overlay on road

# % of dots to recolour once we hit an obstacle
RED_FRAC = 0.3

BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2)


def draw_arrow_right_half(
    img: np.ndarray,
    road_mask: np.ndarray,
    boxes: list[BBox],
    step_px: int = 8,
    min_pts: int = 8,
) -> np.ndarray:
    """Overlay centre-line dots & helper slices on *right* road half."""

    h, _ = road_mask.shape[:2]

    centres: list[tuple[int, int]]   = []  # (x, y)
    slices:  list[tuple[int, int, int]] = []  # (x0, x1, y) for the thin line

    # 1) walk bottom-up collecting candidate x-centres
    for y in range(h - 1, -1, -step_px):
        row_idx = np.flatnonzero(road_mask[y])  # x positions where mask == 255
        if row_idx.size < 3:
            if centres:  # we were in road already → done
                break
            continue

        x_left, x_right = int(row_idx[0]), int(row_idx[-1])
        x_mid = (x_left + x_right) // 2
        right_idx = row_idx[row_idx >= x_mid]
        if right_idx.size < 3:
            continue

        x_c = int((right_idx[0] + right_idx[-1]) // 2)
        centres.append((x_c, y))
        slices.append((int(right_idx[0]), int(right_idx[-1]), y))

    if len(centres) < min_pts:
        return img  # nothing reliable

    # 2) quick mode filter to smooth weird jumps
    xs = np.array([x for x, _ in centres])
    dominant_x = int(np.bincount(xs).argmax())
    centres = [ (dominant_x, y) if abs(x - dominant_x) > 6 else (x, y)
                for (x, y) in centres ]

    # 3) detect first slice that intersects with a bounding box
    touch_idx = None
    for i, (x0, x1, y) in enumerate(slices):
        for (bx1, by1, bx2, by2) in boxes:
            if y >= by1 and y <= by2 and not (x1 < bx1 or x0 > bx2):
                touch_idx = i
                break
        if touch_idx is not None:
            break

    # fix: always set red_start based on fraction if ANY collision occurs
    if touch_idx is not None:
        red_start = int(len(centres) * (1 - RED_FRAC))
    else:
        red_start = len(centres)

    # 4) draw – use red for the last chunk
    for i, ((x, y), (x0, x1, y_slice)) in enumerate(zip(centres, slices)):
        col = RED if i >= red_start else GREEN
        cv2.circle(img, (x, y), 4, col, -1)
        cv2.line(img, (x0, y_slice), (x1, y_slice), col, 1)

    return img

# --- Helpers ---
def iter_pairs(ldir: Path, rdir: Path, pattern: str):
    """yield (left_path, right_path) sorted by name."""
    for lp in sorted(ldir.glob(pattern)):
        rp = rdir / lp.name
        if rp.exists():
            yield lp, rp

def main():
    here = Path(__file__).resolve().parent
    calib = parse_kitti_calib(here / "calibration_KITTI.txt")

    dataset = "training"
    prefix  = "um"  # KITTI sub-set

    root = here.parent / "KITTI"
    ldir = root / "data_road" / dataset / "image_2"
    rdir = root / "data_road_right" / dataset / "image_3"

    for lp, rp in iter_pairs(ldir, rdir, f"{prefix}_*.png"):
        frame = cv2.imread(str(lp))
        if frame is None:
            print(f"could not load {lp.name}")
            continue

        l_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)

        l_crop = crop_bottom_half(l_gray)
        r_crop = crop_bottom_half(r_gray)

        boxes, _, road_mask = detect_obstacles(l_crop, r_crop, frame.shape, calib)

        vis = overlay_mask(frame.copy(), road_mask, ROAD, alpha=0.5)
        draw_bboxes(vis, boxes)
        vis = draw_arrow_right_half(vis, road_mask, boxes)

        cv2.imshow("Arrow Visualization", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
