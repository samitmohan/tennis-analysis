"""Benchmark metrics, as pure functions with no models.

The ball and court rules copy the reference repos named in SPEC.md, so the numbers
compare with theirs: yastrebksv/TrackNet `validate(min_dist=5)` and
yastrebksv/TennisCourtDetector `base_validator.py` (max_dist=7).
"""

import math
import statistics

Point = tuple[float, float] | None


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_ball(preds: list[Point], gts: list[Point], max_dist: float = 5.0) -> dict:
    """Per-frame ball scoring, TrackNet's rule.

    A gt of None means the ball is not visible in that frame. A prediction too far
    from a visible ball counts as a false positive, not a false negative.
    """
    tp = fp = fn = tn = 0
    errors: list[float] = []
    for p, g in zip(preds, gts, strict=True):
        if p is not None and g is not None:
            d = math.dist(p, g)
            if d < max_dist:
                tp += 1
                errors.append(d)
            else:
                fp += 1
        elif p is not None:
            fp += 1
        elif g is not None:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "errors": errors}


def summarize_ball(counts: dict) -> dict:
    out = {k: counts[k] for k in ("tp", "fp", "fn", "tn")}
    out.update(prf(counts["tp"], counts["fp"], counts["fn"]))
    out["median_px"] = statistics.median(counts["errors"]) if counts["errors"] else None
    return out


def add_counts(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def match_events(pred_frames: list[int], gt_frames: list[int], tol: int = 5) -> dict:
    """One-to-one matching of event frames within +-tol, closest pairs first."""
    pairs = sorted(
        (abs(p - g), i, j)
        for i, p in enumerate(pred_frames)
        for j, g in enumerate(gt_frames)
        if abs(p - g) <= tol
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return {"tp": tp, "fp": len(pred_frames) - tp, "fn": len(gt_frames) - tp}


def _in_image(p: Point, width: int, height: int) -> bool:
    return p is not None and 0 <= p[0] <= width and 0 <= p[1] <= height


def score_court(
    preds: list[Point], gts: list[Point], width: int = 1280, height: int = 720,
    max_dist: float = 7.0,
) -> dict:
    """Per-keypoint court scoring, TennisCourtDetector's rule. Lists are flat over all
    keypoints of all images; `dists` keeps every distance where both points are in frame."""
    tp = fp = fn = tn = 0
    dists: list[float] = []
    for p, g in zip(preds, gts, strict=True):
        pin, gin = _in_image(p, width, height), _in_image(g, width, height)
        if pin and gin:
            d = math.dist(p, g)
            dists.append(d)
            if d < max_dist:
                tp += 1
            else:
                fp += 1
        elif pin:
            fp += 1
        elif gin:
            fn += 1
        else:
            tn += 1
    total = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "accuracy": (tp + tn) / total if total else 0.0,
        "median_px": statistics.median(dists) if dists else None,
    }
