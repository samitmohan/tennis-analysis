import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_vid(vid_path: str) -> tuple[list[np.ndarray], float]:
    """Read video frames and return (frames, fps)."""
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    logger.info("Read %d frames at %.1f FPS from %s", len(frames), fps, vid_path)
    return frames, fps


def save_vid(
    output_vid_frames: list[np.ndarray],
    output_vid_path: str,
    fps: float = 24.0,
) -> None:
    """Save frames to video file at the specified FPS."""
    if not output_vid_frames:
        raise ValueError("No frames provided to save video")

    frame_height, frame_width = output_vid_frames[0].shape[:2]
    if any(frame.shape[:2] != (frame_height, frame_width) for frame in output_vid_frames):
        raise ValueError("Inconsistent frame sizes in video")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_vid_path, fourcc, fps, (frame_width, frame_height))

    try:
        for frame in output_vid_frames:
            out.write(frame)
    except Exception as e:
        logger.error("Error writing video: %s", e)
    finally:
        out.release()
    logger.info("Saved %d frames to %s", len(output_vid_frames), output_vid_path)