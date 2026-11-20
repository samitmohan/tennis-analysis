"""TrackNet dataset loader.

Layout: <root>/gameN/ClipM/{0000.jpg, ..., Label.csv}, 1280x720 at 30 fps.
Label.csv columns: file name, visibility (0 = ball not visible), x-coordinate,
y-coordinate, status (0 = flying, 1 = hit, 2 = bounce).
"""

import csv
from dataclasses import dataclass
from pathlib import Path

# Split by game, so frames from one match never sit on both sides.
TRAIN_GAMES = [f"game{i}" for i in range(1, 9)]
TEST_GAMES = ["game9", "game10"]


@dataclass
class Clip:
    game: str
    name: str
    frames: list[Path]
    balls: list[tuple[float, float] | None]
    visibility: list[int]
    hits: list[int]
    bounces: list[int]


def _num(s: str) -> float | None:
    s = s.strip()
    return float(s) if s else None


def load_clips(root: str | Path, games: list[str]) -> list[Clip]:
    root = Path(root)
    clips = []
    for game in games:
        clip_dirs = sorted(
            (d for d in (root / game).iterdir() if d.is_dir()),
            key=lambda d: int(d.name.removeprefix("Clip")),
        )
        for d in clip_dirs:
            with open(d / "Label.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            frames, balls, vis, hits, bounces = [], [], [], [], []
            prev = None
            for i, r in enumerate(rows):
                frames.append(d / r["file name"])
                v = int(r["visibility"])
                x, y = _num(r["x-coordinate"]), _num(r["y-coordinate"])
                vis.append(v)
                balls.append((x, y) if v != 0 and x is not None and y is not None else None)
                status = _num(r["status"])
                # Some events are labelled on two or three consecutive frames: count each once.
                if status != prev:
                    if status == 1:
                        hits.append(i)
                    elif status == 2:
                        bounces.append(i)
                prev = status
            clips.append(Clip(game, d.name, frames, balls, vis, hits, bounces))
    return clips
