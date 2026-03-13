import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .rally import Rally


def export_stats_json(
    output_path: Path | str,
    video_info: dict[str, Any],
    player_summary: dict[str, Any],
    shot_events: list[dict[str, Any]],
    rallies: list[Rally],
) -> None:
    """Export match statistics to a JSON file."""
    data = {
        "video": video_info,
        "player_summary": player_summary,
        "shot_events": shot_events,
        "rallies": [asdict(r) for r in rallies],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_stats_csv(
    output_path: Path | str,
    shot_events: list[dict[str, Any]],
) -> None:
    """Export shot events to a CSV file."""
    if not shot_events:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(shot_events[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shot_events)


def build_video_info(
    input_path: str,
    fps: float,
    total_frames: int,
) -> dict[str, Any]:
    """Build video metadata dict for export."""
    return {
        "input_path": input_path,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(total_frames / fps, 2),
    }


def build_player_summary(
    player_stats_df: "pd.DataFrame",
    player_ids: list[int] = [1, 2],
) -> dict[str, Any]:
    """Build per-player summary from the stats dataframe."""
    import pandas as pd

    summary = {}
    last_row = player_stats_df.iloc[-1]

    for pid in player_ids:
        prefix = f"player_{pid}"
        num_shots = int(last_row.get(f"{prefix}_number_of_shots", 0))
        avg_shot_speed = last_row.get(f"{prefix}_average_shot_speed", 0.0)
        avg_player_speed = last_row.get(f"{prefix}_average_player_speed", 0.0)

        if pd.isna(avg_shot_speed):
            avg_shot_speed = 0.0
        if pd.isna(avg_player_speed):
            avg_player_speed = 0.0

        summary[f"player_{pid}"] = {
            "total_shots": num_shots,
            "avg_shot_speed_kmh": round(float(avg_shot_speed), 1),
            "avg_player_speed_kmh": round(float(avg_player_speed), 1),
        }

    return summary


def build_shot_events(
    ball_shot_frames: list[int],
    fps: float,
    shot_speeds: dict[int, float],
    shot_players: dict[int, int],
) -> list[dict[str, Any]]:
    """Build a list of shot event dicts for export."""
    events = []
    for frame in ball_shot_frames:
        events.append({
            "frame": frame,
            "timestamp_seconds": round(frame / fps, 2),
            "player": shot_players.get(frame),
            "ball_speed_kmh": round(shot_speeds.get(frame, 0.0), 1),
        })
    return events
