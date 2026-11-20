"""Run a trained ball tracker over frames, as a stream or over a clip."""

from collections import deque
from collections.abc import Iterable, Iterator

import cv2
import numpy as np
import torch

from ball_tracking.data import IN_H, IN_W, decode, to_input
from ball_tracking.model import TrackNet

Point = tuple[float, float] | None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt: str, device: str) -> TrackNet:
    model = TrackNet()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device, memory_format=torch.channels_last).eval()


@torch.no_grad()
def stream_predict(
    model: TrackNet, frames: Iterable[np.ndarray], device: str, batch: int = 16, thr: float = 0.5,
) -> Iterator[Point]:
    """One ball position per frame. Holds a 3-frame window of 512x288 copies and one
    batch of inputs, never the video."""
    win: deque[np.ndarray] = deque(maxlen=3)
    pending: list[np.ndarray] = []
    size = None

    def flush() -> list[Point]:
        x = torch.from_numpy(np.stack(pending))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            prob = torch.sigmoid(model(to_input(x, device)).float()).cpu().numpy()
        pending.clear()
        return [decode(p[0], size[0], size[1], thr) for p in prob]

    for frame in frames:
        if size is None:
            size = (frame.shape[1], frame.shape[0])
        small = cv2.resize(frame, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        if not win:
            win.extend([small, small])  # the first frame stands in for its two predecessors
        win.append(small)
        pending.append(np.concatenate(list(win), axis=2).transpose(2, 0, 1))
        if len(pending) == batch:
            yield from flush()
    if pending:
        yield from flush()


def predict_clip(
    model: TrackNet, frames: list[np.ndarray], device: str, batch: int = 16, thr: float = 0.5,
) -> list[Point]:
    return list(stream_predict(model, frames, device, batch, thr))
