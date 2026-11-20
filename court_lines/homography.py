"""Court geometry: the 14 keypoints in metres, and a homography between court and image.

Keypoint order matches the TennisCourtDetector dataset the keypoint model trained on:
0-1 far baseline (left, right), 2-3 near baseline, 4-5 left singles line (far, near),
6-7 right singles line, 8-9 far service line (left, right), 10-11 near service line,
12-13 centre service line (far, near). Court x runs left to right across the doubles
width, y runs from the far baseline to the near one. ITF dimensions.
"""

import cv2
import numpy as np

LENGTH, DOUBLES_W = 23.77, 10.97
ALLEY = 1.37  # doubles sideline to singles sideline
NET_Y = LENGTH / 2
SERVICE = 6.40  # net to service line
L, R = ALLEY, DOUBLES_W - ALLEY
FAR_S, NEAR_S, MID_X = NET_Y - SERVICE, NET_Y + SERVICE, DOUBLES_W / 2

COURT_KPS_M = np.array([
    (0, 0), (DOUBLES_W, 0), (0, LENGTH), (DOUBLES_W, LENGTH),
    (L, 0), (L, LENGTH), (R, 0), (R, LENGTH),
    (L, FAR_S), (R, FAR_S), (L, NEAR_S), (R, NEAR_S),
    (MID_X, FAR_S), (MID_X, NEAR_S),
], dtype=np.float32)


def fit_homography(image_kps: np.ndarray, thresh_px: float = 5.0) -> np.ndarray | None:
    """Court metres -> image pixels, fit with RANSAC so a few bad keypoints get outvoted."""
    pts = np.asarray(image_kps, np.float32).reshape(-1, 2)
    H, inliers = cv2.findHomography(COURT_KPS_M, pts, cv2.RANSAC, thresh_px)
    if H is None or inliers.sum() < 6:
        return None
    return H


def court_to_image(H: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts_m, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def image_to_court(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Valid only for points on the ground plane: feet, bounces. Not a ball in the air."""
    pts = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)


def refine_keypoints(image_kps: np.ndarray, thresh_px: float = 5.0) -> np.ndarray:
    """Replace the 14 predicted points with the court model projected through the
    RANSAC homography. Falls back to the input when no homography fits."""
    H = fit_homography(image_kps, thresh_px)
    return np.asarray(image_kps, np.float32).reshape(-1, 2) if H is None else court_to_image(H, COURT_KPS_M)
