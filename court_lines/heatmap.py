"""Court keypoints as 14 heatmaps instead of 28 regressed numbers.

    python -m court_lines.heatmap --data <court_lines.prep output> --out <run dir>

The tutorial's ResNet50 squeezes the frame to 224x224 and regresses 28 numbers, so one
output unit is 5.7 px wide at 1280x720; it scores accuracy 0.828 at 7 px with a 3.82 px
median on the val split. Here the same U-Net as the ball tracker (one frame in, one
heatmap per keypoint out, 640x360) localises each point on its own grid.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ball_tracking.data import decode
from ball_tracking.model import TrackNet, weighted_bce
from evaluation.metrics import score_court

IN_W, IN_H = 640, 360
SRC_W, SRC_H = 1280, 720
# A horizontal flip swaps left and right: new keypoint i is old keypoint FLIP[i].
FLIP = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]


def inside(kps) -> bool:
    return all(0 < x < SRC_W and 0 < y < SRC_H for x, y in kps)


class CourtHeatmaps(Dataset):
    def __init__(self, root: Path, items: list[dict], radius: float = 4.0, augment: bool = False):
        self.root, self.items, self.radius, self.augment = root, items, radius, augment
        self.xs = np.arange(IN_W, dtype=np.float32)
        self.ys = np.arange(IN_H, dtype=np.float32)[:, None]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        img = cv2.imread(str(self.root / "images" / f"{it['id']}.jpg"))
        kps = np.asarray(it["kps"], np.float32) * (IN_W / SRC_W)
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1]
            kps = kps[FLIP].copy()
            kps[:, 0] = IN_W - 1 - kps[:, 0]
        hm = np.stack([
            (((self.xs - x) ** 2 + (self.ys - y) ** 2) <= self.radius**2).astype(np.float32)
            for x, y in kps
        ])
        x = np.ascontiguousarray(img.transpose(2, 0, 1))
        return torch.from_numpy(x), torch.from_numpy(hm)


def to_input(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.to(device, non_blocking=True).float().div_(255).contiguous(
        memory_format=torch.channels_last
    )


def predict_kps(prob: np.ndarray) -> list[tuple[float, float] | None]:
    """14 x IN_H x IN_W probabilities -> 14 points in 1280x720 pixels."""
    return [decode(p, SRC_W, SRC_H, thr=0.5) for p in prob]


@torch.no_grad()
def validate(model, loader, gts, device) -> dict:
    model.eval()
    preds = []
    for x, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prob = torch.sigmoid(model(to_input(x, device)).float()).cpu().numpy()
        for p in prob:
            preds += predict_kps(p)
    model.train()
    return score_court(preds, gts, SRC_W, SRC_H)


def _worker_init(_: int) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    root, out, device = Path(args.data), Path(args.out), "cuda"
    out.mkdir(parents=True, exist_ok=True)
    train = json.loads((root / "data_train.json").read_text())
    # Checkpoints are picked on 10% of the train split; the val split, which
    # evaluation/court.py reports on, is never seen here.
    order = np.random.default_rng(0).permutation(len(train))
    n_sel = len(train) // 10
    select = [train[i] for i in order[:n_sel] if inside(train[i]["kps"])]
    train = [train[i] for i in order[n_sel:]]
    val, val_gts = select, [tuple(p) for it in select for p in it["kps"]]
    train_dl = DataLoader(
        CourtHeatmaps(root, train, augment=True), batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True,
        worker_init_fn=_worker_init,
    )
    val_dl = DataLoader(
        CourtHeatmaps(root, val), batch_size=16, num_workers=args.workers, pin_memory=True,
        worker_init_fn=_worker_init,
    )

    model = TrackNet(frames=1, out_ch=14).to(device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    log = open(out / "log.jsonl", "w")
    best = -1.0

    for epoch in range(args.epochs):
        t0, losses = time.perf_counter(), []
        for x, hm in train_dl:
            x = to_input(x, device)
            gain = torch.empty(x.shape[0], 1, 1, 1, device=device).uniform_(0.8, 1.2)
            bias = torch.empty(x.shape[0], 1, 1, 1, device=device).uniform_(-0.08, 0.08)
            x = (x * gain + bias).clamp_(0, 1)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
            loss = weighted_bce(logits.float(), hm.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        train_s = time.perf_counter() - t0
        m = validate(model, val_dl, val_gts, device)
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "train_s": train_s,
               "images_per_s": len(losses) * args.bs / train_s, **m}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        state = {"model": model.state_dict(), "epoch": epoch, "val": m}
        torch.save(state, out / "last.pt")
        if m["accuracy"] > best:
            best = m["accuracy"]
            torch.save(state, out / "best.pt")


if __name__ == "__main__":
    main()
