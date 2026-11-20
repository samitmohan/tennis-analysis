"""Side-by-side GIF: the tutorial's YOLO ball detector against the 3-frame tracker on
one held-out TrackNet clip.

    python -m evaluation.demo_ball --data <tracknet root> --preds preds.json \
        --clip game9/Clip2 --out output_videos/ball_compare.gif

The label ring is white, each detector's ball is a filled dot with a short trail. The
panel headers carry that clip's own recall and precision under the 5 px rule.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import json
import numpy as np
from ultralytics import YOLO

from evaluation.ball import center, pick, yolo_boxes
from evaluation.metrics import score_ball, summarize_ball
from evaluation.tracknet import load_clips

PANEL = (640, 360)
TRAIL = 10


def _panel(frame, gt, track, i, title, stats, color):
    img = cv2.resize(frame, PANEL, interpolation=cv2.INTER_AREA)
    s = PANEL[0] / frame.shape[1]
    pts = [None if p is None else (int(p[0] * s), int(p[1] * s)) for p in track[max(0, i - TRAIL):i + 1]]
    for a, b in zip(pts, pts[1:]):
        if a and b:
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)
    if gt[i] is not None:
        cv2.circle(img, (int(gt[i][0] * s), int(gt[i][1] * s)), 11, (255, 255, 255), 1, cv2.LINE_AA)
    if pts[-1]:
        cv2.circle(img, pts[-1], 4, color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (PANEL[0], 46), (20, 20, 20), -1)
    cv2.putText(img, title, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(img, stats, (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--clip", default="game9/Clip2")
    ap.add_argument("--weights", default="models/last.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default="output_videos/ball_compare.gif")
    args = ap.parse_args()

    game, name = args.clip.split("/")
    (clip,) = [c for c in load_clips(args.data, [game]) if c.name == name]
    frames = [cv2.imread(str(f)) for f in clip.frames]
    yolo = [center(pick(b, "best")) for b in yolo_boxes(YOLO(args.weights), frames, args.conf)]
    tracker = [None if p is None else tuple(p) for p in json.loads(Path(args.preds).read_text())["preds"][args.clip]]

    def stats(pred):
        s = summarize_ball(score_ball(pred, clip.balls))
        return f"this clip: recall {s['recall']:.2f}, precision {s['precision']:.2f}"

    left = ("Tutorial YOLO, one frame", stats(yolo), (80, 80, 255))
    right = ("3-frame heatmap tracker (this repo)", stats(tracker), (60, 220, 255))
    print(args.clip, "YOLO", left[1], "| tracker", right[1])

    step = max(1, round(30 / args.fps))
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "demo.mp4"
        w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (PANEL[0] * 2, PANEL[1]))
        for i in range(0, len(frames), step):
            w.write(np.hstack([_panel(frames[i], clip.balls, yolo, i, *left),
                               _panel(frames[i], clip.balls, tracker, i, *right)]))
        w.release()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                        "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        args.out], check=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
