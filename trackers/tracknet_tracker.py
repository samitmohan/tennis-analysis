import logging
import pickle

from ball_tracking.infer import load_model, pick_device, stream_predict
from trackers.ball_tracker import BallTracker

logger = logging.getLogger(__name__)

HALF = 4.0  # px; a small box around the point so the drawing and interpolation code still apply


class TrackNetBallTracker(BallTracker):
    """BallTracker's interface, with positions from the 3-frame heatmap model.

    On TrackNet's test games it finds the ball in far more frames than the single-frame
    YOLO detector; results/ has both numbers.
    """

    def __init__(self, ckpt: str, thr: float = 0.5, batch: int = 16) -> None:
        self.device = pick_device()
        self.model = load_model(ckpt, self.device)
        self.thr, self.batch = thr, batch
        logger.info("TrackNet ball tracker on %s", self.device)

    def detect_frames(self, frames, read_from_stub: bool = False, stub_path: str | None = None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                return pickle.load(f)
        detections = [
            {1: [p[0] - HALF, p[1] - HALF, p[0] + HALF, p[1] + HALF]} if p else {}
            for p in stream_predict(self.model, frames, self.device, self.batch, self.thr)
        ]
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(detections, f)
        return detections
