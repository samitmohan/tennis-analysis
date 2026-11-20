"""Ball and hit benchmark on the TrackNet test games.

    python -m evaluation.ball --data <tracknet root> --detector yolo --weights models/last.pt

One YOLO pass keeps every box, so two selection rules score from the same detections:
`last`, the tutorial's loop (keeps the lowest-confidence box), and `best`, the
highest-confidence box. Hits come from the pipeline's shot detector run on the
interpolated positions of each clip.
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from evaluation.metrics import add_counts, match_events, prf, score_ball, summarize_ball
from evaluation.tracknet import TEST_GAMES, load_clips
from trackers.ball_tracker import BallTracker

Box = tuple[float, float, float, float, float]  # x1, y1, x2, y2, conf


def yolo_boxes(model, frames: list, conf: float, batch: int = 16) -> list[list[Box]]:
    out: list[list[Box]] = []
    for i in range(0, len(frames), batch):
        for res in model.predict(frames[i:i + batch], conf=conf, verbose=False):
            out.append([(*b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes])
    return out


def pick(boxes: list[Box], rule: str) -> Box | None:
    if not boxes:
        return None
    return boxes[-1] if rule == "last" else max(boxes, key=lambda b: b[4])


def center(b: Box | None) -> tuple[float, float] | None:
    return None if b is None else ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def shot_frames(chosen: list[Box | None]) -> list[int]:
    """The pipeline's own shot detector, on its own interpolated positions."""
    if not any(chosen):
        return []
    tracker = BallTracker.__new__(BallTracker)
    positions = [{1: list(b[:4])} if b else {} for b in chosen]
    positions = tracker.interpolate_ball_positions(positions)
    return tracker.get_ball_shot_frames(positions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="models/last.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--out", default="results/ball_yolo.json")
    ap.add_argument("--preds", help="score a ball_tracking.predict JSON instead of running YOLO")
    args = ap.parse_args()

    if args.preds:
        return score_preds(args)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    clips = load_clips(args.data, TEST_GAMES)
    zero_ball = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "errors": []}
    zero_ev = {"tp": 0, "fp": 0, "fn": 0}
    ball = {r: dict(zero_ball) for r in ("last", "best")}
    hits = {r: dict(zero_ev) for r in ("last", "best")}
    n_frames, t_model = 0, 0.0

    for clip in clips:
        frames = [cv2.imread(str(p)) for p in clip.frames]
        t0 = time.perf_counter()
        boxes = yolo_boxes(model, frames, args.conf)
        t_model += time.perf_counter() - t0
        n_frames += len(frames)
        for rule in ("last", "best"):
            chosen = [pick(b, rule) for b in boxes]
            ball[rule] = add_counts(ball[rule], score_ball([center(b) for b in chosen], clip.balls))
            hits[rule] = add_counts(hits[rule], match_events(shot_frames(chosen), clip.hits))

    result = {
        "detector": f"yolo {Path(args.weights).name} conf={args.conf}",
        "games": TEST_GAMES,
        "clips": len(clips),
        "frames": n_frames,
        "model_fps": n_frames / t_model,
        "ball": {r: summarize_ball(ball[r]) for r in ball},
        "hits": {r: {**hits[r], **prf(hits[r]["tp"], hits[r]["fp"], hits[r]["fn"])} for r in hits},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def score_preds(args) -> None:
    """Same ball rule and the same shot detector as the YOLO run, on saved positions."""
    saved = json.loads(Path(args.preds).read_text())
    clips = load_clips(args.data, TEST_GAMES)
    ball = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "errors": []}
    hits = {"tp": 0, "fp": 0, "fn": 0}
    for clip in clips:
        pts = [tuple(p) if p else None for p in saved["preds"][f"{clip.game}/{clip.name}"]]
        ball = add_counts(ball, score_ball(pts, clip.balls))
        boxes = [(p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1, 1.0) if p else None for p in pts]
        hits = add_counts(hits, match_events(shot_frames(boxes), clip.hits))
    result = {
        "detector": f"tracknet {saved['ckpt']} thr={saved['thr']}",
        "games": TEST_GAMES,
        "clips": len(clips),
        "frames": sum(len(c.frames) for c in clips),
        "model_fps": saved["model_fps"],
        "ball": summarize_ball(ball),
        "hits_by_pipeline_shot_detector": {**hits, **prf(hits["tp"], hits["fp"], hits["fn"])},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
