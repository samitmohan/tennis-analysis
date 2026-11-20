"""Shrink the TennisCourtDetector dataset to 640x360 JPEGs, straight from the zip.

    python -m court_lines.prep --zip court_dataset.zip --out <dir>

Writes <out>/images/<id>.jpg and copies data_train.json / data_val.json. Keypoints
stay in 1280x720 coordinates. Decoding from the zip avoids extracting 6.8 GB of PNGs.
"""

import argparse
import json
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

OUT_W, OUT_H = 640, 360
_zip: zipfile.ZipFile | None = None
_out: Path | None = None


def _init(zip_path: str, out: str) -> None:
    global _zip, _out
    _zip, _out = zipfile.ZipFile(zip_path), Path(out)
    cv2.setNumThreads(1)


def _shrink(img_id: str) -> None:
    buf = np.frombuffer(_zip.read(f"data/images/{img_id}.png"), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert img.shape[:2] == (720, 1280), f"{img_id}: {img.shape}"
    small = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(_out / "images" / f"{img_id}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ids = []
    with zipfile.ZipFile(args.zip) as z:
        for split in ("train", "val"):
            data = z.read(f"data/data_{split}.json")
            (out / f"data_{split}.json").write_bytes(data)
            ids += [it["id"] for it in json.loads(data)]
    with Pool(args.workers, initializer=_init, initargs=(args.zip, args.out)) as pool:
        pool.map(_shrink, ids, chunksize=32)
    print(f"{len(ids)} images -> {out}")


if __name__ == "__main__":
    main()
