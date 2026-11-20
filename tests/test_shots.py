"""Shot speed and line calls, checked with a homography whose answers are known."""

import numpy as np
import pytest

from analytics.shots import Shot, in_singles, shots_from_events
from court_lines.homography import court_to_image

H = np.array([[38.0, -9.0, 430.0], [0.0, 12.0, 150.0], [0.0, -0.012, 1.0]])


def _px(x_m, y_m):
    return tuple(court_to_image(H, np.array([[x_m, y_m]], np.float32))[0])


def _bbox_with_feet_at(x_m, y_m):
    fx, fy = _px(x_m, y_m)
    return [fx - 20, fy - 120, fx + 20, fy]


def test_speed_is_ground_distance_over_time():
    # hitter on the near baseline centre, bounce 18 m up the court, 20 frames at 30 fps
    fps, hit, bounce = 30.0, 10, 30
    ball = [None] * 40
    ball[hit] = _px(5.485, 23.0)
    ball[bounce] = _px(5.485, 5.0)
    players = [{} for _ in range(40)]
    players[hit] = {1: _bbox_with_feet_at(5.485, 23.0), 2: _bbox_with_feet_at(5.0, 0.0)}
    (shot,) = shots_from_events([hit], [bounce], ball, players, H, fps)
    assert shot.player == 1 and shot.call == "in"
    assert shot.speed_kmh == pytest.approx(18.0 / (20 / 30) * 3.6, rel=1e-3)  # 97.2 km/h


def test_bounce_after_next_hit_is_not_paired():
    ball = [None] * 50
    for f in (5, 20, 30):
        ball[f] = _px(5.0, 10.0)
    players = [{1: _bbox_with_feet_at(5.0, 22.0)} for _ in range(50)]
    shots = shots_from_events([5, 20], [30], ball, players, H, 30.0)
    assert shots[0].bounce_frame is None and shots[1].bounce_frame == 30


def test_line_calls():
    assert in_singles(1.37, 0.0) and in_singles(9.60, 23.77)  # on the lines
    assert not in_singles(1.2, 12.0) and not in_singles(5.0, 24.0)


def test_stats_count_last_shot_and_skip_untimed_speeds():
    from pipeline import _build_stats_dataframe, _stats_from_shots

    shots = [Shot(3, 1, None, None, None, None), Shot(10, 1, 40, (5.0, 1.0), 80.0, "in"),
             Shot(50, 2, 70, (5.0, 20.0), 60.0, "in"), Shot(90, 1, 110, (5.0, 2.0), 100.0, "in")]
    players = [{1: [0, 0, 2, 2], 2: [0, 0, 2, 2]} for _ in range(120)]
    df = _build_stats_dataframe(_stats_from_shots(shots, players, np.eye(3), 30.0), 120)
    last = df.iloc[-1]
    assert last["player_1_number_of_shots"] == 3  # the last shot counts too
    assert abs(last["player_1_average_shot_speed"] - 90.0) < 1e-9  # mean of 80 and 100; the untimed hit is not 0
    assert abs(last["player_2_average_shot_speed"] - 60.0) < 1e-9
