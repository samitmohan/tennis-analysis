"""Train the ball tracker on TrackNet games 1-7; pick the checkpoint on game 8.

    python -m ball_tracking.train --data <tracknet root> --out <run dir>

The test games (9 and 10) are never loaded here.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ball_tracking.data import BallFrames, decode, to_input
from ball_tracking.model import TrackNet, weighted_bce
from evaluation.metrics import score_ball, summarize_ball
from evaluation.tracknet import TRAIN_GAMES, load_clips

FIT_GAMES, VAL_GAMES = TRAIN_GAMES[:7], TRAIN_GAMES[7:]
FRAME_W, FRAME_H = 1280, 720  # every TrackNet frame


def _worker_init(_: int) -> None:
    # Each worker would otherwise start 16 OpenCV threads on a 16-core box, and ten
    # workers starve the process that launches the GPU kernels.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


@torch.no_grad()
def validate(model, loader, balls, device) -> dict:
    model.eval()
    preds = []
    for x, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prob = torch.sigmoid(model(to_input(x, device)).float())
        preds += [decode(p[0], FRAME_W, FRAME_H) for p in prob.cpu().numpy()]
    model.train()
    return summarize_ball(score_ball(preds, balls))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--shrunk", action="store_true", help="--data holds prep.py's 512x288 frames")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True  # fixed input size, so tune conv algorithms once
    np.random.seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    fit = load_clips(args.data, FIT_GAMES)
    val = load_clips(args.data, VAL_GAMES)
    val_balls = [b for c in val for b in c.balls]
    src = (FRAME_W, FRAME_H) if args.shrunk else None
    train_dl = DataLoader(
        BallFrames(fit, augment=True, src_size=src), batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True,
        worker_init_fn=_worker_init,
    )
    val_dl = DataLoader(
        BallFrames(val, src_size=src), batch_size=16, num_workers=args.workers, pin_memory=True,
        worker_init_fn=_worker_init,
    )

    model = TrackNet().to(device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    log = open(out / "log.jsonl", "a")
    best_f1, step = -1.0, 0

    for epoch in range(args.epochs):
        t0, losses, t_warm = time.perf_counter(), [], None
        for x, hm in train_dl:
            x = to_input(x, device)
            # Per-sample brightness and contrast jitter, done here instead of in the workers.
            gain = torch.empty(x.shape[0], 1, 1, 1, device=device).uniform_(0.8, 1.2)
            bias = torch.empty(x.shape[0], 1, 1, 1, device=device).uniform_(-0.08, 0.08)
            x = (x * gain + bias).clamp_(0, 1)
            hm = hm.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
            loss = weighted_bce(logits.float(), hm)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if len(losses) == 20:
                t_warm = time.perf_counter()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        train_s = time.perf_counter() - t0
        metrics = validate(model, val_dl, val_balls, device)
        rec = {
            "epoch": epoch, "step": step, "loss": float(np.mean(losses)),
            "samples_per_s": len(losses) * args.bs / train_s, "train_s": train_s,
            "steady_samples_per_s": (len(losses) - 20) * args.bs / (time.perf_counter() - t_warm)
            if t_warm and len(losses) > 20 else None,
            "val_f1": metrics["f1"], "val_precision": metrics["precision"],
            "val_recall": metrics["recall"], "val_median_px": metrics["median_px"],
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        state = {"model": model.state_dict(), "epoch": epoch, "val": metrics, "args": vars(args)}
        torch.save(state, out / "last.pt")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(state, out / "best.pt")
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
