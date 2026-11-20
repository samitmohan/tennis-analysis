import cv2
import numpy as np
import torch

from ball_tracking.infer import pick_device
from ball_tracking.model import TrackNet
from court_lines.court_line_detect import CourtLineDetect
from court_lines.heatmap import IN_H, IN_W, predict_kps, to_input
from court_lines.homography import COURT_KPS_M, court_to_image


class CourtHeatmapDetect(CourtLineDetect):
    """CourtLineDetect's interface (predict -> 28 numbers), from the 14-heatmap U-Net.

    On the TennisCourtDetector val split it scores accuracy 0.991 at 7 px against the
    ResNet50's 0.828 (results/court_*.json). A keypoint whose heatmap stays below the
    threshold is filled in from the RANSAC homography of the others.
    """

    def __init__(self, ckpt: str) -> None:
        self.device = pick_device()
        model = TrackNet(frames=1, out_ch=14)
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.model = model.to(self.device, memory_format=torch.channels_last).eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        small = cv2.resize(image, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(small.transpose(2, 0, 1)))[None]
        prob = torch.sigmoid(self.model(to_input(x, self.device)).float()).cpu().numpy()[0]
        pts = predict_kps(prob)  # in 1280x720
        kps = np.array([p if p is not None else (np.nan, np.nan) for p in pts], np.float32)
        kps *= np.array([w / 1280, h / 720], np.float32)
        # Return the court model projected through a RANSAC homography: it fills keypoints
        # whose heatmap stayed dark and replaces outliers (on val: accuracy 0.9913 -> 0.9927).
        found = ~np.isnan(kps[:, 0])
        if found.sum() < 6:
            raise ValueError(f"court keypoints not found ({int((~found).sum())} of 14 missing)")
        H, inliers = cv2.findHomography(COURT_KPS_M[found], kps[found], cv2.RANSAC, 5.0)
        if H is None or inliers.sum() < 6:
            raise ValueError("court keypoints do not fit a court")
        return court_to_image(H, COURT_KPS_M).reshape(-1)
