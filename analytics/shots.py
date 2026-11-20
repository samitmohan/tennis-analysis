"""Shot speed and line calls from ground contacts only.

The original pipeline projected every ball position onto the court with one
metres-per-pixel scale taken from a player's height. Two things break that: depth is
foreshortened in a broadcast view, so one scale cannot serve both axes, and a ball in
the air does not sit above its court position. Here only two kinds of point are
projected through the court homography: players' feet and ball bounces, both on the
ground plane. Shot speed is the hitter's feet at the hit to the bounce, over the time
between them: an average speed, lower than the radar's peak off the racket.
"""

from dataclasses import dataclass

import numpy as np

from court_lines.homography import ALLEY, DOUBLES_W, LENGTH, image_to_court


@dataclass
class Shot:
    frame: int
    player: int | None
    bounce_frame: int | None
    bounce_m: tuple[float, float] | None
    speed_kmh: float | None
    call: str | None  # "in" / "out" against the singles court, None without a bounce


def feet_px(bbox: list[float]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def to_court(H: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    x, y = image_to_court(H, np.array([pt], np.float32))[0]
    return float(x), float(y)


def in_singles(x_m: float, y_m: float, margin: float = 0.0) -> bool:
    """Line balls are in: the line belongs to the court."""
    return (ALLEY - margin <= x_m <= DOUBLES_W - ALLEY + margin) and (-margin <= y_m <= LENGTH + margin)


def shots_from_events(
    hits: list[int], bounces: list[int], ball_px: list[tuple[float, float] | None],
    players: list[dict[int, list[float]]], H: np.ndarray, fps: float,
) -> list[Shot]:
    """One Shot per hit, paired with the first bounce before the next hit."""
    out = []
    for k, hit in enumerate(hits):
        nxt = hits[k + 1] if k + 1 < len(hits) else len(ball_px)
        bounce = next((b for b in bounces if hit < b < nxt), None)
        player, feet_m = None, None
        if players[hit] and ball_px[hit] is not None:
            bx, by = ball_px[hit]
            player = min(players[hit], key=lambda pid: np.hypot(
                feet_px(players[hit][pid])[0] - bx, feet_px(players[hit][pid])[1] - by))
            feet_m = to_court(H, feet_px(players[hit][player]))
        bounce_m = speed = call = None
        if bounce is not None and ball_px[bounce] is not None:
            bounce_m = to_court(H, ball_px[bounce])
            call = "in" if in_singles(*bounce_m) else "out"
            if feet_m is not None:
                dist = float(np.hypot(bounce_m[0] - feet_m[0], bounce_m[1] - feet_m[1]))
                speed = dist / ((bounce - hit) / fps) * 3.6
        out.append(Shot(hit, player, bounce, bounce_m, speed, call))
    return out
