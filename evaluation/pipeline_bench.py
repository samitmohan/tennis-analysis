"""End-to-end pipeline wall time and peak RAM, under GNU time.

    python -m evaluation.pipeline_bench --old-dir <checkout of the tutorial-era code> \
        --videos input_videos/input_video.mp4 input_videos/input_x5.mp4

Runs three configurations per video: the old pipeline (whole video in RAM), this
pipeline with the tutorial's YOLO ball detector, and this pipeline with the heatmap
tracker and homography. Linux only (needs /usr/bin/time -v). Writes
results/pipeline_bench.json; --logs <dir> parses an earlier run's logs instead.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2

MODES = {
    "old": "Old pipeline (whole video in RAM, YOLO ball)",
    "new_yolo": "This pipeline, tutorial YOLO ball, heatmap court",
    "new_tracknet": "This pipeline, heatmap ball + heatmap court + homography",
}


def parse_time_log(text: str) -> dict:
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text).group(1)
    secs = sum(float(p) * 60**i for i, p in enumerate(reversed(wall.split(":"))))
    rss_kb = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1))
    ok = "Traceback" not in text and re.search(r"Exit status: 0", text) is not None
    return {"wall_s": secs, "maxrss_gb": rss_kb / 1024**2, "ok": ok}


def run(mode: str, video: Path, out: Path, old_dir: Path | None) -> str:
    py = sys.executable
    if mode == "old":
        cmd = [py, "-I", "-c", "import sys; sys.path.insert(0, '.'); "
               f"sys.argv=['cli','-i','{video.resolve()}','-o','{out}']; import cli; cli.main()"]
        cwd = old_dir
    else:
        cmd = [py, "cli.py", "-i", str(video), "-o", str(out)]
        if mode == "new_yolo":
            cfg = out.parent / "yolo.yaml"
            cfg.write_text('models:\n  ball_tracker: "yolo"\n')
            cmd += ["-c", str(cfg)]
        cwd = None
    res = subprocess.run(["/usr/bin/time", "-v", *cmd], cwd=cwd, capture_output=True, text=True)
    return res.stdout + res.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["input_videos/input_video.mp4"])
    ap.add_argument("--old-dir", type=Path)
    ap.add_argument("--work", type=Path, default=Path("bench_runs"))
    ap.add_argument("--logs", type=Path, help="parse <mode>_<video stem>.log files from here")
    ap.add_argument("--out", default="results/pipeline_bench.json")
    args = ap.parse_args()

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() or "CPU"
    runs = []
    for video in map(Path, args.videos):
        frames = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
        for mode, label in MODES.items():
            if args.logs:
                text = (args.logs / f"{mode}_{video.stem}.log").read_text()
            else:
                if mode == "old" and args.old_dir is None:
                    continue
                args.work.mkdir(parents=True, exist_ok=True)
                text = run(mode, video, args.work / f"{mode}_{video.stem}", args.old_dir)
                (args.work / f"{mode}_{video.stem}.log").write_text(text)
            runs.append({"label": f"{label}, {video.name}", "mode": mode, "video": video.name,
                         "frames": frames, **parse_time_log(text)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"gpu": gpu, "runs": runs}, indent=2))
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
