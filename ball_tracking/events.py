"""Hit and bounce detection from a ball trajectory.

    python -m ball_tracking.events --data <tracknet root> --preds preds.json --out results/events.json

The pipeline's original shot detector looks for a sign change in vertical image motion,
held for 25 frames. It finds some hits and no bounces. Here a small MLP reads the
trajectory around each frame (positions relative to the current one, +-W frames) and
scores flying / hit / bounce. It trains on the tracker's own predictions for games
1-7, so it sees the same noise it meets at inference; game 8 picks the thresholds;
games 9-10 are scored once.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluation.metrics import match_events, prf
from evaluation.tracknet import TEST_GAMES, TRAIN_GAMES, Clip, load_clips

W = 7  # frames each side; 233 ms at 30 fps
FIT, VAL = TRAIN_GAMES[:7], TRAIN_GAMES[7:]
Point = tuple[float, float] | None


def fill_gaps(points: list[Point], max_gap: int = 6) -> np.ndarray:
    """Linear interpolation across short gaps; longer gaps stay NaN."""
    xy = np.array([p if p is not None else (np.nan, np.nan) for p in points], np.float32)
    seen = ~np.isnan(xy[:, 0])
    idx = np.flatnonzero(seen)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            t = np.linspace(0, 1, b - a + 1)[1:-1, None]
            xy[a + 1:b] = xy[a] * (1 - t) + xy[b] * t
    return xy


def features(points: list[Point], own_y: bool = False) -> np.ndarray:
    """Per frame: dx, dy (in 100 px) to each frame within +-W, a seen flag for each,
    and whether the frame itself has a position. own_y adds the frame's y / 720; v1
    used it, and it ties the model to one broadcast's framing."""
    xy = fill_gaps(points)
    n = len(xy)
    pad = np.full((W, 2), np.nan, np.float32)
    xyp = np.concatenate([pad, xy, pad])
    rows = []
    for t in range(n):
        win = xyp[t:t + 2 * W + 1]
        rel = (win - xy[t]) / 100.0
        seen = ~np.isnan(rel[:, 0])
        rel = np.nan_to_num(rel)
        extra = [float(seen[W])]
        if own_y:
            extra.insert(0, 0.0 if np.isnan(xy[t, 1]) else xy[t, 1] / 720.0)
        rows.append(np.concatenate([rel.ravel(), seen.astype(np.float32), extra]))
    return np.stack(rows).astype(np.float32)


def mirror(x: np.ndarray) -> np.ndarray:
    """The same trajectory seen in a mirror: every dx changes sign."""
    m = x.copy()
    m[:, 0:2 * (2 * W + 1):2] *= -1
    return m


def labels(clip: Clip, spread: int = 1) -> np.ndarray:
    """0 flying, 1 hit, 2 bounce; each event also labels +-spread frames around it."""
    y = np.zeros(len(clip.frames), np.int64)
    for cls, frames in ((1, clip.hits), (2, clip.bounces)):
        for f in frames:
            y[max(0, f - spread):f + spread + 1] = cls
    return y


class EventNet(nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pick(prob: np.ndarray, thr: float, nms: int = 5) -> list[int]:
    """Frames where prob peaks above thr, at least nms frames apart, strongest first."""
    order = np.argsort(-prob)
    taken: list[int] = []
    for i in order:
        if prob[i] < thr:
            break
        if all(abs(i - j) > nms for j in taken):
            taken.append(int(i))
    return sorted(taken)


def _dataset(clips, preds, own_y):
    xs = [features(preds[f"{c.game}/{c.name}"], own_y) for c in clips]
    ys = [labels(c) for c in clips]
    return xs, ys


def _train(xs, ys, epochs: int, augment: bool) -> EventNet:
    torch.manual_seed(0)
    X, Y = np.concatenate(xs), np.concatenate(ys)
    if augment:
        X, Y = np.concatenate([X, mirror(X)]), np.concatenate([Y, Y])
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    model = EventNet(X.shape[1])
    counts = torch.bincount(Y, minlength=3).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (3 * counts))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            loss = loss_fn(model(X[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def _probs(model, xs) -> list[np.ndarray]:
    with torch.no_grad():
        return [torch.softmax(model(torch.from_numpy(x)), 1).numpy() for x in xs]


HIT_NMS = (5, 10)  # frames; ties go to 5, the bounce window


def _score(probs, clips, thr_hit, thr_bounce, hit_nms=5):
    out = {"hits": {"tp": 0, "fp": 0, "fn": 0}, "bounces": {"tp": 0, "fp": 0, "fn": 0}}
    for p, c in zip(probs, clips):
        for key, cls, thr, nms, gt in (("hits", 1, thr_hit, hit_nms, c.hits),
                                       ("bounces", 2, thr_bounce, 5, c.bounces)):
            m = match_events(pick(p[:, cls], thr, nms), gt, tol=5)
            out[key] = {k: out[key][k] + m[k] for k in m}
    return {k: {**v, **prf(v["tp"], v["fp"], v["fn"])} for k, v in out.items()}


def _thresholds(probs, clips) -> dict[str, float]:
    # Plain floats: numpy scalars in the saved bundle fail torch.load(weights_only=True).
    grid = [round(float(t), 2) for t in np.arange(0.3, 0.96, 0.05)]
    # Hit suppression window, chosen with the threshold: merged hit labels sit 11+ frames
    # apart, so a second hit 6-10 frames after the first is a tracker artefact.
    hit, hit_nms = max(((t, n) for n in HIT_NMS for t in grid),
                       key=lambda tn: _score(probs, clips, tn[0], 0.5, tn[1])["hits"]["f1"])
    return {
        "hit": hit, "hit_nms": hit_nms,
        "bounce": max(grid, key=lambda t: _score(probs, clips, 0.5, t)["bounces"]["f1"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="results/events.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--save", default="models/events.pt")
    ap.add_argument("--own-y", action="store_true", help="v1 feature: the frame's image y")
    ap.add_argument("--cv", action="store_true",
                    help="pick thresholds by leave-one-game-out over games 1-8, train on all 8")
    ap.add_argument("--no-mirror", action="store_true", help="v1: no mirrored trajectories")
    args = ap.parse_args()
    augment = not args.no_mirror

    preds = json.loads(Path(args.preds).read_text())["preds"]
    test = load_clips(args.data, TEST_GAMES)
    tx, _ = _dataset(test, preds, args.own_y)
    result = {"preds": args.preds, "window_frames": W, "own_y": args.own_y,
              "mirror": augment, "cv": args.cv}

    if args.cv:
        # Out-of-fold probabilities for each training game, from a model that never saw it.
        oof_probs, oof_clips = [], []
        for g in TRAIN_GAMES:
            held = load_clips(args.data, [g])
            rest = load_clips(args.data, [x for x in TRAIN_GAMES if x != g])
            rx, ry = _dataset(rest, preds, args.own_y)
            hx, _ = _dataset(held, preds, args.own_y)
            oof_probs += _probs(_train(rx, ry, args.epochs, augment), hx)
            oof_clips += held
        thr = _thresholds(oof_probs, oof_clips)
        result["oof_games_1_8"] = _score(oof_probs, oof_clips, thr["hit"], thr["bounce"], thr["hit_nms"])
        ax, ay = _dataset(load_clips(args.data, TRAIN_GAMES), preds, args.own_y)
        model = _train(ax, ay, args.epochs, augment)
    else:
        # v1: train on games 1-7, thresholds from game 8 alone.
        fx, fy = _dataset(load_clips(args.data, FIT), preds, args.own_y)
        val = load_clips(args.data, VAL)
        vx, _ = _dataset(val, preds, args.own_y)
        model = _train(fx, fy, args.epochs, augment)
        thr = _thresholds(_probs(model, vx), val)
        result["val_game_8"] = _score(_probs(model, vx), val, thr["hit"], thr["bounce"], thr["hit_nms"])

    result["thresholds"] = thr
    result["test"] = _score(_probs(model, tx), test, thr["hit"], thr["bounce"], thr["hit_nms"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    n_in = model.net[0].in_features
    torch.save(
        {"model": model.state_dict(), "n_in": n_in, "own_y": args.own_y, "thresholds": thr},
        args.save,
    )
    print(json.dumps(result, indent=2))


class EventDetector:
    """Loads what main() saves; returns (hit frames, bounce frames) for a trajectory."""

    def __init__(self, path: str) -> None:
        bundle = torch.load(path, map_location="cpu")
        self.model = EventNet(bundle["n_in"])
        self.model.load_state_dict(bundle["model"])
        self.model.eval()
        self.thr = bundle["thresholds"]
        self.own_y = bundle.get("own_y", True)  # bundles without the key came from v1

    def __call__(self, points: list[Point]) -> tuple[list[int], list[int]]:
        if not points:
            return [], []
        with torch.no_grad():
            x = torch.from_numpy(features(points, self.own_y))
            p = torch.softmax(self.model(x), 1).numpy()
        return pick(p[:, 1], self.thr["hit"], self.thr.get("hit_nms", 5)), pick(p[:, 2], self.thr["bounce"])


if __name__ == "__main__":
    main()
