"""Tennis match analysis pipeline.

Orchestrates player detection, ball tracking, court keypoint detection,
speed calculations, rally detection, and visualization.
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from config import Config, load_config
from trackers import PlayerTracker, BallTracker
from court_lines import CourtLineDetect
from mini_court import MiniCourt
from utils import (
    iter_frames,
    probe,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters,
)
from analytics.rally import detect_rallies
from analytics.export import (
    export_stats_json,
    export_stats_csv,
    build_video_info,
    build_player_summary,
    build_shot_events,
)
from analytics.heatmap import generate_heatmap
from analytics.shots import Shot, feet_px, shots_from_events, to_court
from ball_tracking.events import EventDetector
from court_lines.heatmap_detect import CourtHeatmapDetect
from court_lines.homography import fit_homography
from trackers.tracknet_tracker import TrackNetBallTracker

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: str,
    output_dir: str,
    config: Config | None = None,
    use_stubs: bool = False,
    stubs_dir: str = "tracker_stubs",
    export_stats: str | None = None,
    generate_heatmaps: bool = False,
    no_video: bool = False,
) -> dict[str, Any]:
    """Run the full tennis analysis pipeline.

    Args:
        input_path: Path to input video file.
        output_dir: Directory for output files.
        config: Pipeline configuration (uses defaults if None).
        use_stubs: Use cached detection stubs instead of running models.
        stubs_dir: Directory containing detection stub files.
        export_stats: Path to export stats JSON (None to skip).
        generate_heatmaps: Whether to generate player heatmap PNGs.
        no_video: Skip output video generation.

    Returns:
        Dict containing pipeline results (stats dataframe, rallies, etc.).
    """
    if config is None:
        config = load_config()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Read video: streamed, one frame in memory at a time ---
    logger.info("Reading video: %s", input_path)
    fps, _ = probe(input_path)
    first_frame = next(iter_frames(input_path))

    # --- Detection ---
    player_tracker = PlayerTracker(config.models.player_detector)
    if config.models.ball_tracker == "tracknet":
        ball_tracker = TrackNetBallTracker(config.models.tracknet)
    else:
        ball_tracker = BallTracker(
            config.models.ball_detector, confidence=config.models.ball_confidence
        )

    player_detections = player_tracker.detect_frames(
        iter_frames(input_path),
        read_from_stub=use_stubs,
        stub_path=f"{stubs_dir}/player_detections.pkl" if use_stubs else None,
    )
    ball_detections = ball_tracker.detect_frames(
        iter_frames(input_path),
        read_from_stub=use_stubs,
        stub_path=f"{stubs_dir}/ball_detections.pkl" if use_stubs else None,
    )
    # Raw positions, before interpolation: the event model reads the gaps too.
    ball_points = [
        ((d[1][0] + d[1][2]) / 2, (d[1][1] + d[1][3]) / 2) if d.get(1) else None
        for d in ball_detections
    ]
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    n_frames = len(player_detections)
    logger.info("Detected over %d frames at %.1f FPS", n_frames, fps)

    # --- Court keypoints ---
    if config.models.court_model == "heatmap":
        court_line_detect = CourtHeatmapDetect(config.models.court_heatmap)
    else:
        court_line_detect = CourtLineDetect(config.models.court_keypoint)
    court_keypoints = court_line_detect.predict(first_frame)
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    mini_court_obj = MiniCourt(first_frame)
    H = fit_homography(np.asarray(court_keypoints, np.float32).reshape(14, 2))
    use_geometry = (
        config.models.ball_tracker == "tracknet"
        and H is not None
        and Path(config.models.events).exists()
    )
    shots: list[Shot] = []

    if use_geometry:
        # --- Ground-contact geometry: events from the trajectory, metres from the homography ---
        hits, bounces = EventDetector(config.models.events)(ball_points)
        player_detections = _renumber_players(player_detections)
        shots = shots_from_events(hits, bounces, ball_points, player_detections, H, fps)
        ball_shot_frames = hits
        shot_speeds = {s.frame: s.speed_kmh for s in shots if s.speed_kmh is not None}
        shot_players = {s.frame: s.player for s in shots if s.player is not None}
        player_stats_data = _stats_from_shots(shots, player_detections, H, fps)
        player_mini_court_detections, ball_mini_court_detections = _mini_court_positions(
            mini_court_obj, player_detections, shots, H, n_frames, fps
        )
        logger.info(
            "%d hits, %d bounces, %d shots with a speed",
            len(hits), len(bounces), len(shot_speeds),
        )
    else:
        logger.warning(
            "Using the tutorial's geometry (ball_tracker=%s, homography %s, event model %s)",
            config.models.ball_tracker, "found" if H is not None else "missing",
            config.models.events if Path(config.models.events).exists() else "missing",
        )
        ball_shot_frames = ball_tracker.get_ball_shot_frames(
            ball_detections,
            rolling_window=config.shot_detection.rolling_window,
            minimum_change_frames=config.shot_detection.minimum_change_frames,
        )
        player_mini_court_detections, ball_mini_court_detections = (
            mini_court_obj.convert_bounding_boxes_to_mini_court_coordinates(
                player_detections, ball_detections, court_keypoints
            )
        )
        player_stats_data, shot_speeds, shot_players = _calculate_stats(
            ball_shot_frames,
            ball_mini_court_detections,
            player_mini_court_detections,
            mini_court_obj,
            config,
            fps,
        )
    player_stats_data_df = _build_stats_dataframe(player_stats_data, n_frames)

    # --- Rally detection ---
    rallies = detect_rallies(
        ball_shot_frames,
        fps=fps,
        gap_threshold_seconds=config.rally.gap_threshold_seconds,
        shot_speeds=shot_speeds,
        shot_players=shot_players,
    )
    logger.info("Detected %d rallies from %d shots", len(rallies), len(ball_shot_frames))

    # --- Export statistics ---
    if export_stats:
        video_info = build_video_info(input_path, fps, n_frames)
        player_summary = build_player_summary(player_stats_data_df)
        shot_events = build_shot_events(ball_shot_frames, fps, shot_speeds, shot_players)
        by_frame = {s.frame: s for s in shots}
        for ev in shot_events:
            s = by_frame.get(ev["frame"])
            if s is not None and s.bounce_m is not None:
                ev["bounce_frame"] = s.bounce_frame
                ev["bounce_court_m"] = [round(v, 2) for v in s.bounce_m]
                ev["call"] = s.call

        stats_path = Path(export_stats)
        export_stats_json(stats_path, video_info, player_summary, shot_events, rallies)
        logger.info("Exported stats JSON to %s", stats_path)

        csv_path = stats_path.with_suffix(".csv")
        export_stats_csv(csv_path, shot_events)
        logger.info("Exported shot events CSV to %s", csv_path)

    # --- Heatmaps ---
    if generate_heatmaps:
        court_bounds = (
            mini_court_obj.court_start_x,
            mini_court_obj.court_start_y,
            mini_court_obj.court_end_x,
            mini_court_obj.court_end_y,
        )
        for player_id in [1, 2]:
            heatmap_path = str(output_path / f"heatmap_player_{player_id}.png")
            generate_heatmap(
                player_mini_court_detections,
                court_bounds,
                heatmap_path,
                player_id=player_id,
            )
            logger.info("Generated heatmap: %s", heatmap_path)

    # --- Output video: a second streamed pass, drawing and writing one frame at a time ---
    if not no_video:
        video_output_path = str(output_path / "output_video.mp4")
        h, w = first_frame.shape[:2]
        writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        try:
            for i, frame in enumerate(iter_frames(input_path)):
                f = player_tracker.draw_boxes([frame], [player_detections[i]])
                f = ball_tracker.draw_boxes(f, [ball_detections[i]])
                f = court_line_detect.draw_keypts_vid(f, court_keypoints)
                f = mini_court_obj.draw_mini_court(f)
                f = mini_court_obj.draw_points_on_mini_court(f, [player_mini_court_detections[i]])
                f = mini_court_obj.draw_points_on_mini_court(
                    f, [ball_mini_court_detections[i]], color=(0, 255, 255)
                )
                # draw_player_stats indexes frames by the dataframe's index
                f = draw_player_stats(f, player_stats_data_df.iloc[[i]].reset_index(drop=True))
                cv2.putText(
                    f[0], f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                writer.write(f[0])
        finally:
            writer.release()
        logger.info("Saved output video: %s", video_output_path)

    return {
        "fps": fps,
        "total_frames": n_frames,
        "ball_shot_frames": ball_shot_frames,
        "player_stats_df": player_stats_data_df,
        "rallies": rallies,
        "shot_speeds": shot_speeds,
        "shot_players": shot_players,
        "player_mini_court_detections": player_mini_court_detections,
        "shots": shots,
    }


def _renumber_players(dets: list[dict[int, list[float]]]) -> list[dict[int, list[float]]]:
    """Stats columns are player_1 / player_2; tracker ids are arbitrary. Map the two
    chosen ids to 1 and 2, in increasing id order."""
    ids = sorted({pid for d in dets for pid in d})
    remap = {pid: i + 1 for i, pid in enumerate(ids[:2])}
    return [{remap[pid]: box for pid, box in d.items() if pid in remap} for d in dets]


def _stats_from_shots(
    shots: list[Shot], players: list[dict[int, list[float]]], H: np.ndarray, fps: float,
) -> list[dict]:
    """The same cumulative rows _calculate_stats builds, with speeds in metres from
    ground contacts: shot speed from Shot, opponent speed from feet between hits."""
    rows = [{"frame_num": 0, **{
        f"player_{p}_{k}": 0 for p in (1, 2) for k in (
            "number_of_shots", "timed_shots", "total_shot_speed", "last_shot_speed",
            "total_player_speed", "last_player_speed",
        )
    }}]
    for k, s in enumerate(shots):
        if s.player not in (1, 2):
            continue
        row = deepcopy(rows[-1])
        row["frame_num"] = s.frame
        row[f"player_{s.player}_number_of_shots"] += 1
        if s.speed_kmh is not None:  # a hit with no bounce found has no speed, not 0 km/h
            row[f"player_{s.player}_timed_shots"] += 1
            row[f"player_{s.player}_total_shot_speed"] += s.speed_kmh
            row[f"player_{s.player}_last_shot_speed"] = s.speed_kmh
        opp = 2 if s.player == 1 else 1
        nxt = shots[k + 1] if k + 1 < len(shots) else None
        a = players[s.frame].get(opp)
        b = players[nxt.frame].get(opp) if nxt else None
        if a and b:
            pa, pb = to_court(H, feet_px(a)), to_court(H, feet_px(b))
            v = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1])) / ((nxt.frame - s.frame) / fps) * 3.6
            row[f"player_{opp}_total_player_speed"] += v
            row[f"player_{opp}_last_player_speed"] = v
        rows.append(row)
    return rows


def _mini_court_positions(mini, players, shots: list[Shot], H, n_frames: int, fps: float):
    """Players' feet every frame; the ball only where it touched the ground, held on
    screen for one second after each bounce."""
    def to_mini(xy_m):
        return (
            mini.court_start_x + mini.convert_meters_to_pixels(xy_m[0]),
            mini.court_start_y + mini.convert_meters_to_pixels(xy_m[1]),
        )

    player_pos = [
        {pid: to_mini(to_court(H, feet_px(box))) for pid, box in d.items()} for d in players
    ]
    ball_pos: list[dict] = [{} for _ in range(n_frames)]
    for s in shots:
        if s.bounce_frame is not None and s.bounce_m is not None:
            for f in range(s.bounce_frame, min(n_frames, s.bounce_frame + int(fps))):
                ball_pos[f] = {1: to_mini(s.bounce_m)}
    return player_pos, ball_pos


def _calculate_stats(
    ball_shot_frames: list[int],
    ball_mini_court_detections: list[dict],
    player_mini_court_detections: list[dict],
    mini_court_obj: MiniCourt,
    config: Config,
    fps: float,
) -> tuple[list[dict], dict[int, float], dict[int, int]]:
    """Calculate per-shot speed statistics.

    Returns:
        (player_stats_data, shot_speeds, shot_players) where:
        - player_stats_data: list of cumulative stat dicts per shot
        - shot_speeds: mapping of shot frame -> ball speed in km/h
        - shot_players: mapping of shot frame -> player id who hit the shot
    """
    shot_speeds: dict[int, float] = {}
    shot_players: dict[int, int] = {}

    player_stats_data = [
        {
            "frame_num": 0,
            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0,
            "player_1_last_shot_speed": 0,
            "player_1_total_player_speed": 0,
            "player_1_last_player_speed": 0,
            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0,
            "player_2_last_shot_speed": 0,
            "player_2_total_player_speed": 0,
            "player_2_last_player_speed": 0,
        }
    ]

    for ball_shot_idx in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_idx]
        end_frame = ball_shot_frames[ball_shot_idx + 1]
        # BUG FIX: use actual video FPS instead of hardcoded 24
        ball_shot_time_in_seconds = (end_frame - start_frame) / fps

        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            config.court.double_line_width,
            mini_court_obj.get_width_of_mini_court(),
        )
        speed_of_ball = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id],
                ball_mini_court_detections[start_frame][1],
            ),
        )

        shot_speeds[start_frame] = speed_of_ball
        shot_players[start_frame] = player_shot_ball

        # Opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id],
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            config.court.double_line_width,
            mini_court_obj.get_width_of_mini_court(),
        )
        speed_of_opponent = (
            distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
        )

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += speed_of_ball
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = speed_of_ball
        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += (
            speed_of_opponent
        )
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = (
            speed_of_opponent
        )

        player_stats_data.append(current_player_stats)

    return player_stats_data, shot_speeds, shot_players


def _build_stats_dataframe(
    player_stats_data: list[dict], total_frames: int
) -> pd.DataFrame:
    """Build a per-frame stats dataframe with averages computed correctly."""
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({"frame_num": list(range(total_frames))})
    player_stats_data_df = pd.merge(
        frames_df, player_stats_data_df, on="frame_num", how="left"
    )
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df["player_1_average_shot_speed"] = (
        player_stats_data_df["player_1_total_shot_speed"]
        / player_stats_data_df.get("player_1_timed_shots", player_stats_data_df["player_1_number_of_shots"])
    )
    player_stats_data_df["player_2_average_shot_speed"] = (
        player_stats_data_df["player_2_total_shot_speed"]
        / player_stats_data_df.get("player_2_timed_shots", player_stats_data_df["player_2_number_of_shots"])
    )
    # BUG FIX: each player's avg speed now divides by their OWN shot count
    player_stats_data_df["player_1_average_player_speed"] = (
        player_stats_data_df["player_1_total_player_speed"]
        / player_stats_data_df["player_1_number_of_shots"]
    )
    player_stats_data_df["player_2_average_player_speed"] = (
        player_stats_data_df["player_2_total_player_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )

    return player_stats_data_df
