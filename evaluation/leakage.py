"""Check whether a training image set contains frames from the TrackNet games.

    python -m evaluation.leakage --images training/tennis-ball-detection-6 --data <tracknet root>

A 64-bit difference hash survives resizing and re-encoding, so a Roboflow export of
a broadcast frame still lands within a few bits of the original. Reports, per game,
how many training images sit within --max-bits of some frame of that game.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.tracknet import TEST_GAMES, TRAIN_GAMES, load_clips


def dhash(path: Path) -> int:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-bits", type=int, default=6)
    ap.add_argument("--out", default="results/leakage.json")
    args = ap.parse_args()

    train_imgs = sorted(p for p in Path(args.images).rglob("*.jpg"))
    a = np.array([dhash(p) for p in train_imgs], dtype=np.uint64)
    report = {"training_images": len(train_imgs), "max_bits": args.max_bits, "games": {}}
    for game in TRAIN_GAMES + TEST_GAMES:
        frames = [f for c in load_clips(args.data, [game]) for f in c.frames]
        b = np.array([dhash(p) for p in frames], dtype=np.uint64)
        x = a[:, None] ^ b[None, :]
        bits = np.unpackbits(x.view(np.uint8).reshape(len(a), len(b), 8), axis=2).sum(axis=2)
        hits = int((bits.min(axis=1) <= args.max_bits).sum())
        report["games"][game] = {"frames": len(frames), "training_images_matching": hits}
        print(game, len(frames), hits, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
