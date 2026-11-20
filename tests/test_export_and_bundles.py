"""Regressions found by running the pipeline end to end on bqa3."""

import json

import numpy as np
import torch

from analytics.export import export_stats_json
from ball_tracking.events import EventDetector, EventNet, W


def test_export_writes_numpy_float32_speeds(tmp_path):
    out = tmp_path / "stats.json"
    events = [{"frame": 3, "ball_speed_kmh": np.float32(97.2), "player": np.int64(1)}]
    export_stats_json(out, {"fps": 30.0}, {}, events, [])
    data = json.loads(out.read_text())
    assert abs(data["shot_events"][0]["ball_speed_kmh"] - 97.2) < 1e-4


def test_event_bundle_loads_with_weights_only(tmp_path):
    n_in = 3 * (2 * W + 1) + 1
    path = tmp_path / "events.pt"
    torch.save({"model": EventNet(n_in).state_dict(), "n_in": n_in, "own_y": False,
                "thresholds": {"hit": 0.75, "bounce": 0.85}}, path)
    det = EventDetector(str(path))
    hits, bounces = det([(float(i), 300.0 - 2 * i) for i in range(60)])
    assert isinstance(hits, list) and isinstance(bounces, list)


def test_csv_export_handles_shots_with_and_without_bounces(tmp_path):
    from analytics.export import export_stats_csv

    out = tmp_path / "shots.csv"
    export_stats_csv(out, [
        {"frame": 3, "ball_speed_kmh": 90.0},
        {"frame": 40, "ball_speed_kmh": 80.0, "bounce_frame": 52, "call": "in"},
    ])
    lines = out.read_text().splitlines()
    assert lines[0] == "frame,ball_speed_kmh,bounce_frame,call"
    assert lines[1] == "3,90.0,," and lines[2] == "40,80.0,52,in"
