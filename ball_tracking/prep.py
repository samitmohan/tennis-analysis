"""Shrink TrackNet frames to the network's input size once, before training.

    python -m ball_tracking.prep --data <tracknet root> --out <small root>

Decoding three 1280x720 JPEGs per sample kept training at ~30 samples/s while the GPU
alone does ~60. A 512x288 JPEG decodes several times faster and needs no resize.
Labels stay in 1280x720 coordinates; train with --src-size 1280x720.
"""

import argparse
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2

from ball_tracking.data import IN_H, IN_W


def _shrink(job: tuple[Path, Path]) -> None:
    src, dst = job
    img = cv2.imread(str(src))
    assert img.shape[:2] == (720, 1280), f"{src}: {img.shape}"
    small = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    src_root, dst_root = Path(args.data), Path(args.out)
    jobs = []
    for label in src_root.glob("game*/Clip*/Label.csv"):
        dst_dir = dst_root / label.parent.relative_to(src_root)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(label, dst_dir / "Label.csv")
        jobs += [(p, dst_dir / p.name) for p in label.parent.glob("*.jpg")]
    cv2.setNumThreads(1)
    with Pool(args.workers) as pool:
        pool.map(_shrink, jobs, chunksize=64)
    print(f"{len(jobs)} frames -> {dst_root}")


if __name__ == "__main__":
    main()
