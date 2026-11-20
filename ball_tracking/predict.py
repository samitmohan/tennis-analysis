"""Run a trained ball tracker over TrackNet games and save the positions.

    python -m ball_tracking.predict --data <full-res tracknet root> --ckpt best.pt --out preds.json

Output: {"game9/Clip1": [[x, y] or null, ...], ...} in 1280x720 pixels, plus the
model's frames per second. The benchmark and the event model both read this file.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from ball_tracking.infer import load_model, predict_clip
from evaluation.tracknet import TEST_GAMES, TRAIN_GAMES, load_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--games", default="all", help="'all', 'test', or comma-separated")
    args = ap.parse_args()

    games = {"all": TRAIN_GAMES + TEST_GAMES, "test": TEST_GAMES}.get(args.games)
    games = games or args.games.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    preds, n, t_model = {}, 0, 0.0
    for clip in load_clips(args.data, games):
        frames = [cv2.imread(str(p)) for p in clip.frames]
        t0 = time.perf_counter()
        pts = predict_clip(model, frames, device, thr=args.thr)
        t_model += time.perf_counter() - t0
        n += len(frames)
        preds[f"{clip.game}/{clip.name}"] = [list(p) if p else None for p in pts]
    out = {"ckpt": args.ckpt, "thr": args.thr, "frames": n, "model_fps": n / t_model, "preds": preds}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"{n} frames, {n / t_model:.1f} fps (resize + model + decode) -> {args.out}")


if __name__ == "__main__":
    main()
