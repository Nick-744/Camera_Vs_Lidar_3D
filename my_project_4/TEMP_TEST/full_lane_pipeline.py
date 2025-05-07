# full_lane_pipeline.py
"""
Complete prototype implementation of the paper
“Robust Lane Detection Using Multiple Features” in ONE file.

Content
-------
• LaneGradientFeatureExtractor   (§ II‑A‑1)
• LaneIntensityFeatureExtractor  (§ II‑A‑2)
• LaneTextureFeatureExtractor    (§ II‑A‑3)
• FeatureFusion                  (§ II‑C)
• LaneModelEstimator             (§ II‑B)
• KITTI demo loop (run with `python full_lane_pipeline.py`)

The code is stitched together from the individual skeletons discussed
earlier, with **only two fixes** applied:

1. `LaneIntensityFeatureExtractor` now guarantees an *odd* kernel
   height for `GaussianBlur`, per OpenCV requirements.
2. `run_demo()` builds the KITTI path correctly and warns if empty.

Set the environment variable `KITTI_PATH` to your KITTI root directory
(or place a “KITTI” folder next to this file) before running.
"""

from __future__ import annotations
# ─── Standard / 3rd‑party ────────────────────────────────────────────
import os
from pathlib import Path
from time import time
from typing import Callable, Optional, Tuple, List

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# § II‑A‑1  Gradient‑based features
# ─────────────────────────────────────────────────────────────────────
class LaneGradientFeatureExtractor:
    def __init__(self, score_threshold: float = 50.0, horizon_tol: int = 2):
        self._lsd = cv2.createLineSegmentDetector(0)
        self.score_threshold = score_threshold
        self.horizon_tol = horizon_tol

    def extract(self, bgr: np.ndarray, *, homography: np.ndarray | None = None):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        raw = self._detect_line_segments(gray)
        if raw.size == 0:
            return raw, np.empty(0, np.float32)

        horizon_y = (
            self._horizon_from_homography(homography, gray.shape)
            if homography is not None
            else gray.shape[0] // 2
        )
        scores = self._score_segments(raw, horizon_y)
        keep = scores >= self.score_threshold
        return raw[keep], scores

    def _detect_line_segments(self, gray: np.ndarray):
        lines, *_ = self._lsd.detect(gray)
        return np.zeros((0, 4), np.float32) if lines is None else lines.reshape(-1, 4)

    def _horizon_from_homography(self, H: np.ndarray, shape: tuple[int, int]) -> int:
        p = np.array([[0, 0, 1]], dtype=np.float64).T
        hp = H @ p
        hp /= hp[2]
        return int(round(hp[1, 0]))

    def _score_segments(self, segs: np.ndarray, horizon_y: int):
        n = len(segs)
        scores = np.zeros(n, np.float32)
        lines = []
        lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        for x1, y1, x2, y2 in segs:
            a, b = y1 - y2, x2 - x1
            c = x1 * y2 - x2 * y1
            lines.append((a, b, c))
        for i, (ai, bi, ci) in enumerate(lines):
            for j, (aj, bj, cj) in enumerate(lines):
                if i == j:
                    continue
                D = ai * bj - aj * bi
                if abs(D) < 1e-6:
                    continue
                y_int = (-ai * cj + aj * ci) / D
                if abs(y_int - horizon_y) <= self.horizon_tol:
                    scores[i] += lengths[j]
        return scores


# ─────────────────────────────────────────────────────────────────────
# § II‑A‑2  Intensity‑based features
# ─────────────────────────────────────────────────────────────────────
class LaneIntensityFeatureExtractor:
    def __init__(
        self,
        sigma: float = 3.0,
        orientations: tuple[float, ...] = (-15.0, -7.5, 0.0, 7.5, 15.0),
        low_thresh: int = 30,
        high_thresh: int = 80,
        min_length: int = 40,
        ipm_size: tuple[int, int] = (640, 480),
    ):
        self.sigma = sigma
        self.orientations = orientations
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.min_length = min_length
        self.ipm_size = ipm_size

    def extract(
        self,
        bgr: np.ndarray,
        *,
        homography: np.ndarray | None = None,
        return_response: bool = False,
    ):
        top = self._inverse_perspective(bgr, homography)

        # Fix: ensure kernel height is odd
        vert_kernel = (1, max(3, int(6 * self.sigma + 1) | 1))
        smoothed = cv2.GaussianBlur(top, vert_kernel, 0, sigmaY=self.sigma)

        response = self._second_derivative_max(smoothed)
        mask = self._hysteresis_filter(response)

        ys, xs = np.where(mask)
        coords = np.vstack((xs, ys)).T.astype(np.int32)
        return (coords, response) if return_response else (coords, None)

    # helpers
    def _inverse_perspective(self, img_bgr: np.ndarray, H: np.ndarray | None):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if H is None:
            return gray
        H_inv = np.linalg.inv(H)
        w, h = self.ipm_size
        return cv2.warpPerspective(gray, H_inv, (w, h), flags=cv2.INTER_LINEAR)

    def _second_derivative_max(self, gray: np.ndarray):
        h, w = gray.shape
        responses: list[np.ndarray] = []
        ksize = max(3, int(6 * self.sigma + 1) | 1)
        for angle in self.orientations:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)
            rot_blur = cv2.GaussianBlur(rot, (ksize, 0), sigmaX=self.sigma)
            d2 = cv2.Sobel(rot_blur, cv2.CV_32F, 2, 0, ksize=3)
            resp = cv2.convertScaleAbs(d2)
            resp_inv = cv2.warpAffine(
                resp, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
            )
            responses.append(resp_inv)
        return np.maximum.reduce(responses)

    def _hysteresis_filter(self, resp: np.ndarray):
        strong = resp >= self.high_thresh
        weak = (resp >= self.low_thresh) & ~strong
        mask = strong.astype(np.uint8) * 255
        n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(
            weak.astype(np.uint8), connectivity=8
        )
        for lbl in range(1, n_lbl):
            comp = lbl_img == lbl
            if stats[lbl, cv2.CC_STAT_WIDTH] < self.min_length and \
               stats[lbl, cv2.CC_STAT_HEIGHT] < self.min_length:
                continue
            if (strong & comp).any():
                mask[comp] = 255
        return mask


# ─────────────────────────────────────────────────────────────────────
# § II‑A‑3  Texture‑based features
# ─────────────────────────────────────────────────────────────────────
def _naive_grey_segmentor(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    return (s < 40) & (v > 50)


class LaneTextureFeatureExtractor:
    def __init__(self, segmentor: Callable[[np.ndarray], np.ndarray] = _naive_grey_segmentor,
                 min_road_area: int = 10_000):
        self.segmentor = segmentor
        self.min_road_area = min_road_area

    def extract(self, bgr: np.ndarray, *, return_mask: bool = False):
        road_mask = self.segmentor(bgr).astype(np.uint8)
        cleaned = self._largest_component(road_mask)
        if cleaned is None:
            empty = np.empty((0, 2), np.int32)
            return (empty, np.zeros_like(road_mask)) if return_mask else (empty, None)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        boundaries = cv2.morphologyEx(cleaned, cv2.MORPH_GRADIENT, kernel)
        ys, xs = np.where(boundaries > 0)
        coords = np.vstack((xs, ys)).T.astype(np.int32)
        return (coords, cleaned.astype(bool)) if return_mask else (coords, None)

    def _largest_component(self, mask: np.ndarray):
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_lbl <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_id = 1 + areas.argmax()
        if areas.max() < self.min_road_area:
            return None
        return (lbl == max_id).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# § II‑C     Feature fusion
# ─────────────────────────────────────────────────────────────────────
class FeatureFusion:
    def __init__(self, weight_gradient: float = 2.0, weight_intensity: float = 1.5,
                 weight_texture: float = 1.0, conf_thresh: float = 2.5, min_blob: int = 120):
        self.wg, self.wi, self.wt = weight_gradient, weight_intensity, weight_texture
        self.conf_thresh, self.min_blob = conf_thresh, min_blob

    def fuse(
        self,
        img_shape: Tuple[int, int],
        *,
        gradient_segments: Tuple[np.ndarray, np.ndarray] | None,
        intensity_coords: Optional[np.ndarray],
        texture_coords: Optional[np.ndarray],
        return_conf: bool = False,
    ):
        h, w = img_shape
        conf = np.zeros((h, w), np.float32)

        # gradient
        if gradient_segments is not None:
            segs, scores = gradient_segments
            if len(segs):
                s_norm = (scores - scores.min()) / max(scores.ptp(), 1e-6)
                for (x1, y1, x2, y2), s in zip(segs.astype(int), s_norm):
                    cv2.line(conf, (x1, y1), (x2, y2),
                             color=float(self.wg * s), thickness=1)

        # intensity
        if intensity_coords is not None and len(intensity_coords):
            xs, ys = intensity_coords.T
            conf[ys, xs] += self.wi

        # texture
        if texture_coords is not None and len(texture_coords):
            xs, ys = texture_coords.T
            conf[ys, xs] += self.wt

        mask = conf >= self.conf_thresh
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        for i in range(1, n_lbl):
            if stats[i, cv2.CC_STAT_AREA] < self.min_blob:
                mask[lbl == i] = False
        return (mask, conf) if return_conf else (mask, None)


# ─────────────────────────────────────────────────────────────────────
# § II‑B     Lane‑model estimation
# ─────────────────────────────────────────────────────────────────────
class LaneModel:
    def __init__(self, coeffs: np.ndarray, inlier_mask: np.ndarray):
        self.coeffs, self.inlier_mask = coeffs, inlier_mask


class LaneModelEstimator:
    ORDER = 3

    def __init__(self, max_iter: int = 500, dist_thresh: float = .25,
                 min_inliers: int = 40, random_state: int | None = None):
        self.max_iter, self.dist_thresh, self.min_inliers = max_iter, dist_thresh, min_inliers
        self.rng = np.random.default_rng(random_state)

    def estimate(self, pts: np.ndarray) -> Optional[LaneModel]:
        if pts.shape[0] < self.min_inliers:
            return None
        best_mask, best_err = None, np.inf
        for _ in range(self.max_iter):
            idx = self.rng.choice(pts.shape[0], self.ORDER + 1, replace=False)
            coeffs = self._poly_from_samples(pts[idx])
            if coeffs is None:                  # singular sample – skip
                continue
            errs = self._point_errors(pts, coeffs)
            mask = errs < self.dist_thresh
            n_in = mask.sum()
            if n_in < self.min_inliers:
                continue
            mean_err = errs[mask].mean()
            if (best_mask is None or
                n_in > best_mask.sum() or
               (n_in == best_mask.sum() and mean_err < best_err)):
                best_mask, best_err = mask, mean_err
        if best_mask is None:
            return None
        coeffs = self._least_squares(pts[best_mask])
        return LaneModel(coeffs.astype(np.float32), best_mask)

    # helpers
    def _poly_from_samples(self, samples: np.ndarray):
        """
        Fit polynomial through ORDER+1 samples.
        Returns None if the Vandermonde matrix is singular
        (happens when two samples share the same y), so the caller
        can pick another random set instead of crashing.
        """
        y = samples[:, 1]
        A = np.vstack([y**p for p in range(self.ORDER, -1, -1)]).T
        try:
            return np.linalg.solve(A, samples[:, 0])
        except np.linalg.LinAlgError:            # singular matrix
            return None

    def _point_errors(self, pts: np.ndarray, coeffs: np.ndarray):
        x_pred = np.polyval(coeffs, pts[:, 1])
        return np.abs(pts[:, 0] - x_pred)

    def _least_squares(self, pts: np.ndarray):
        y = pts[:, 1]
        A = np.vstack([y**p for p in range(self.ORDER, -1, -1)]).T
        coeffs, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
        return coeffs

# ─────────────────────────────────────────────────────────────────────
# KITTI demo
# ─────────────────────────────────────────────────────────────────────
def run_demo():
    # ── user‑configurable ───────────────────────────────────────────
    DATASET_ROOT = Path(os.getenv("KITTI_PATH",
                         Path(__file__).resolve().parent / '..' / "KITTI")).resolve()
    SPLIT, SEQUENCE, MAX_FRAMES = "testing", "um", 10
    H_IPM = None          # provide homography if available
    # ────────────────────────────────────────────────────────────────

    img_dir = DATASET_ROOT / "data_road" / SPLIT / "image_2"
    files = sorted(img_dir.glob(f"{SEQUENCE}_*.png"))
    if not files:
        raise FileNotFoundError(
            f"No images in {img_dir}. Set KITTI_PATH correctly.")
    files = files[:MAX_FRAMES]

    grad_ex = LaneGradientFeatureExtractor(score_threshold=20.)
    int_ex = LaneIntensityFeatureExtractor(sigma=2.5, low_thresh=20, high_thresh=40)
    tex_ex = LaneTextureFeatureExtractor()
    fuser = FeatureFusion(conf_thresh=1, min_blob=50)
    modelest = LaneModelEstimator(dist_thresh=.2, max_iter=1_000, random_state=0)

    for img_path in files:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Cannot open {img_path}")
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        t0 = time()
        g_lines, g_scores = grad_ex.extract(bgr, homography=H_IPM)
        i_coords, _ = int_ex.extract(bgr, homography=H_IPM)
        t_coords, _ = tex_ex.extract(bgr)
        mask, conf = fuser.fuse(gray.shape,
                                gradient_segments=(g_lines, g_scores),
                                intensity_coords=i_coords,
                                texture_coords=t_coords,
                                return_conf=True)
        ys, xs = np.where(mask)
        model = modelest.estimate(np.vstack((xs, ys)).T.astype(np.float32))
        dt = (time() - t0)
        status = "OK" if model else "FAIL"
        print(f"{img_path.name}: {len(xs)} pts → model {status} ({dt:.1f} sec)")

        # quick visual sanity
        if model:
            disp = cv2.normalize(conf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            h = disp.shape[0]
            ys_vis = np.linspace(0, h - 1, 200)
            xs_vis = np.polyval(model.coeffs, ys_vis)
            for x, y in zip(xs_vis, ys_vis):
                if 0 <= x < disp.shape[1]:
                    disp[int(y), int(round(x))] = (0, 0, 255)
            cv2.imshow("conf + model", cv2.resize(disp, None, fx=.5, fy=.5))
            # Wait for keypress to close the window
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
