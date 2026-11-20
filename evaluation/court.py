"""Court keypoint benchmark on TennisCourtDetector's val split.

    python -m evaluation.court --data <dataset root with data_val.json and images/>

Runs the pipeline's own CourtLineDetect on every val image whose 14 keypoints all lie
inside the 1280x720 frame (the reference repo's filter), then scores with its rule.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from court_lines import CourtLineDetect
from court_lines.homography import refine_keypoints
from evaluation.metrics import score_court

W, H = 1280, 720


def inside(kps: list[list[float]]) -> bool:
    return all(0 < x < W and 0 < y < H for x, y in kps)


def _predictor(args):
    """img (BGR, any size) -> 14 x 2 keypoints in 1280x720 pixels; NaN where none found."""
    if not args.heatmap:
        model = CourtLineDetect(args.weights)

        def predict(img):
            return np.asarray(model.predict(img), np.float32).reshape(14, 2) * (
                np.array([W, H], np.float32) / np.array([img.shape[1], img.shape[0]], np.float32)
            )

        return predict, f"{Path(args.weights).name} (ResNet50 regression, 224x224)"

    import torch

    from ball_tracking.infer import pick_device
    from ball_tracking.model import TrackNet
    from court_lines.heatmap import IN_H, IN_W, predict_kps, to_input

    device = pick_device()
    model = TrackNet(frames=1, out_ch=14)
    model.load_state_dict(torch.load(args.heatmap, map_location="cpu")["model"])
    model = model.to(device, memory_format=torch.channels_last).eval()

    @torch.no_grad()
    def predict(img):
        small = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(small.transpose(2, 0, 1)))[None]
        prob = torch.sigmoid(model(to_input(x, device)).float()).cpu().numpy()[0]
        pts = predict_kps(prob)
        return np.array([p if p is not None else (np.nan, np.nan) for p in pts], np.float32)

    return predict, f"{Path(args.heatmap).name} (U-Net, 14 heatmaps at {IN_W}x{IN_H})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="models/keypointsModel.pth")
    ap.add_argument("--out", default="results/court_baseline.json")
    ap.add_argument("--heatmap", help="checkpoint from court_lines.heatmap instead of the ResNet50")
    args = ap.parse_args()

    root = Path(args.data)
    items = json.loads((root / "data_val.json").read_text())
    kept = [it for it in items if inside(it["kps"])]
    predict, name = _predictor(args)

    raw, refined, gts = [], [], []
    t0 = time.perf_counter()
    for it in kept:
        png, jpg = root / "images" / f"{it['id']}.png", root / "images" / f"{it['id']}.jpg"
        img = cv2.imread(str(png if png.exists() else jpg))
        kp = predict(img)
        raw += [tuple(map(float, p)) if np.isfinite(p).all() else None for p in kp]
        refined += [tuple(map(float, p)) for p in refine_keypoints(np.nan_to_num(kp, nan=-1e4))]
        gts += [tuple(p) for p in it["kps"]]
    elapsed = time.perf_counter() - t0

    result = {
        "model": name,
        "val_images": len(items),
        "scored_images": len(kept),
        "keypoints": len(gts),
        "images_per_s": len(kept) / elapsed,
        "raw": score_court(raw, gts, W, H),
        "homography_ransac_5px": score_court(refined, gts, W, H),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
