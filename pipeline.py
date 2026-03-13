"""Tennis match analysis pipeline.

Orchestrates player detection, ball tracking, court keypoint detection,
speed calculations, rally detection, and visualization.
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from config import Config, load_config
from trackers import PlayerTracker, BallTracker
from court_lines import CourtLineDetect
from mini_court import MiniCourt
from utils import (
    read_vid,
    save_vid,
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

    # --- Read video ---
    logger.info("Reading video: %s", input_path)
    video_frames, fps = read_vid(input_path)
    logger.info("Loaded %d frames at %.1f FPS", len(video_frames), fps)

    # --- Detection ---
    player_tracker = PlayerTracker(config.models.player_detector)
    ball_tracker = BallTracker(
        config.models.ball_detector, confidence=config.models.ball_confidence
    )

    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=f"{stubs_dir}/player_detections.pkl" if use_stubs else None,
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=f"{stubs_dir}/ball_detections.pkl" if use_stubs else None,
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # --- Court keypoints ---
    court_line_detect = CourtLineDetect(config.models.court_keypoint)
    court_keypoints = court_line_detect.predict(video_frames[0])
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    # --- Mini court coordinate conversion ---
    mini_court_obj = MiniCourt(video_frames[0])
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

    # --- Speed calculations ---
    player_stats_data, shot_speeds, shot_players = _calculate_stats(
        ball_shot_frames,
        ball_mini_court_detections,
        player_mini_court_detections,
        mini_court_obj,
        config,
        fps,
    )
    player_stats_data_df = _build_stats_dataframe(player_stats_data, len(video_frames))

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
        video_info = build_video_info(input_path, fps, len(video_frames))
        player_summary = build_player_summary(player_stats_data_df)
        shot_events = build_shot_events(ball_shot_frames, fps, shot_speeds, shot_players)

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

    # --- Output video ---
    if not no_video:
        output_video_frames = player_tracker.draw_boxes(video_frames, player_detections)
        output_video_frames = ball_tracker.draw_boxes(output_video_frames, ball_detections)
        output_video_frames = court_line_detect.draw_keypts_vid(
            output_video_frames, court_keypoints
        )
        output_video_frames = mini_court_obj.draw_mini_court(output_video_frames)
        output_video_frames = mini_court_obj.draw_points_on_mini_court(
            output_video_frames, player_mini_court_detections
        )
        output_video_frames = mini_court_obj.draw_points_on_mini_court(
            output_video_frames, ball_mini_court_detections, color=(0, 255, 255)
        )
        output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

        for i, frame in enumerate(output_video_frames):
            cv2.putText(
                frame, f"Frame: {i}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )

        video_output_path = str(output_path / "output_video.mp4")
        save_vid(output_video_frames, video_output_path, fps=fps)
        logger.info("Saved output video: %s", video_output_path)

    return {
        "fps": fps,
        "total_frames": len(video_frames),
        "ball_shot_frames": ball_shot_frames,
        "player_stats_df": player_stats_data_df,
        "rallies": rallies,
        "shot_speeds": shot_speeds,
        "shot_players": shot_players,
        "player_mini_court_detections": player_mini_court_detections,
    }


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
        / player_stats_data_df["player_1_number_of_shots"]
    )
    player_stats_data_df["player_2_average_shot_speed"] = (
        player_stats_data_df["player_2_total_shot_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
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
