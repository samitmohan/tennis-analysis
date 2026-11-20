"""Training samples and heatmap decoding for the TrackNet-style ball tracker."""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from evaluation.tracknet import Clip

IN_W, IN_H = 512, 288


def stack_frames(frames: list[np.ndarray]) -> np.ndarray:
    """BGR frames (any size) -> uint8 9 x IN_H x IN_W. Stays uint8 until it reaches the
    GPU: a float32 batch of 8 is 42 MB to move between processes, uint8 is 10.6 MB."""
    small = [cv2.resize(f, (IN_W, IN_H), interpolation=cv2.INTER_AREA) for f in frames]
    return np.concatenate(small, axis=2).transpose(2, 0, 1)


def to_input(x: torch.Tensor, device: str) -> torch.Tensor:
    """uint8 batch from stack_frames -> float [0, 1] on the device, channels-last."""
    x = x.to(device, non_blocking=True).float().div_(255)
    return x.contiguous(memory_format=torch.channels_last)


def disk(x: float, y: float, radius: float) -> np.ndarray:
    """Binary target, TrackNetV2's recipe. A sigma-2 Gaussian left ~12 pixels near 1.0
    and the network settled on a flat 0.12 everywhere (peak 0.27 after one epoch)."""
    xs = np.arange(IN_W, dtype=np.float32)
    ys = np.arange(IN_H, dtype=np.float32)[:, None]
    return (((xs - x) ** 2 + (ys - y) ** 2) <= radius**2).astype(np.float32)


def gaussian(x: float, y: float, sigma: float) -> np.ndarray:
    xs = np.arange(IN_W, dtype=np.float32)
    ys = np.arange(IN_H, dtype=np.float32)[:, None]
    return np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2)).astype(np.float32)


def window(i: int, n: int = 3) -> list[int]:
    """Frame indices ending at i; the clip's first frame repeats at the start."""
    return [max(j, 0) for j in range(i - n + 1, i + 1)]


class BallFrames(Dataset):
    def __init__(
        self, clips: list[Clip], radius: float = 3.0, augment: bool = False,
        src_size: tuple[int, int] | None = None,
    ) -> None:
        # src_size = the (w, h) the labels refer to, when frames were shrunk by prep.py
        self.src_size = src_size
        self.items = [(c, i) for c in clips for i in range(len(c.frames))]
        self.radius = radius
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int):
        clip, i = self.items[k]
        frames = [cv2.imread(str(clip.frames[j])) for j in window(i)]
        w, h = self.src_size or (frames[0].shape[1], frames[0].shape[0])
        x = stack_frames(frames)
        ball = clip.balls[i]
        hm = np.zeros((IN_H, IN_W), np.float32)
        if ball is not None:
            hm = disk(ball[0] * IN_W / w, ball[1] * IN_H / h, self.radius)
        # Brightness jitter happens on the GPU (train.py); only the flip happens here.
        if self.augment and np.random.rand() < 0.5:
            x, hm = x[:, :, ::-1], hm[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(
            np.ascontiguousarray(hm)
        )[None]


def decode(prob: np.ndarray, frame_w: int, frame_h: int, thr: float = 0.5):
    """Heatmap probabilities (any H x W) -> blob centre in frame pixels, or None.

    Takes the connected blob that holds the peak and returns its probability-weighted
    centroid, so a second bright blob elsewhere cannot pull the estimate.
    """
    if prob.max() < thr:
        return None
    mask = (prob >= thr).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask)
    peak = np.unravel_index(int(prob.argmax()), prob.shape)
    blob = labels == labels[peak]
    ys, xs = np.nonzero(blob)
    wts = prob[ys, xs]
    cx, cy = float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())
    h_in, w_in = prob.shape
    return (cx * frame_w / w_in, cy * frame_h / h_in)
