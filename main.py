from utils import (
    read_vid,
    save_vid,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters,
)
import constants
import cv2
from trackers import PlayerTracker, BallTracker
from court_lines import CourtLineDetect
from mini_court import MiniCourt
from copy import deepcopy
import pandas as pd


def main():
    inp_vid_path = "input/input_video.mp4"
    video_frames = read_vid(inp_vid_path)
    # detect players
    player_tracker = PlayerTracker("yolov8x")
    # detect balls
    ball_tracker = BallTracker("models/last.pt")

    # draw bounding box on top of frames
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl",
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl"
    )

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # court line detector model
    court_model_path = "models/keypointsModel.pth"
    court_line_detect = CourtLineDetect(court_model_path)
    court_keypoints = court_line_detect.predict(video_frames[0])

    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    # draw mini court
    mini_court_obj = MiniCourt(video_frames[0])

    # detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # conv posn to mini court posn
    player_mini_court_detections, ball_mini_court_detections = (
        mini_court_obj.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )
    )

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
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court_obj.get_width_of_mini_court(),
        )
        # speed
        speed_of_ball = (
            distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
        )
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_mini_court_detections[start_frame][1]
            ),
        )

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id],
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court_obj.get_width_of_mini_court(),
        )

        speed_of_opponent = (
            distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
        )

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += (
            speed_of_ball
        )
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = (
            speed_of_ball
        )

        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += (
            speed_of_opponent
        )
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = (
            speed_of_opponent
        )

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({"frame_num": list(range(len(video_frames)))})
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
    player_stats_data_df["player_1_average_player_speed"] = (
        player_stats_data_df["player_1_total_player_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_2_average_player_speed"] = (
        player_stats_data_df["player_2_total_player_speed"]
        / player_stats_data_df["player_1_number_of_shots"]
    )

    # draw output
    output_video_frames = player_tracker.draw_boxes(
        video_frames, player_detections
    )  # player draw
    output_video_frames = ball_tracker.draw_boxes(
        output_video_frames, ball_detections
    )  # ball draw

    # Ensure keypoints are drawn last
    output_video_frames = court_line_detect.draw_keypts_vid(
        output_video_frames, court_keypoints
    )  # key points draw
    # draw mini court
    output_video_frames = mini_court_obj.draw_mini_court(output_video_frames)
    output_video_frames = mini_court_obj.draw_points_on_mini_court(
        output_video_frames, player_mini_court_detections
    )
    output_video_frames = mini_court_obj.draw_points_on_mini_court(
        output_video_frames, ball_mini_court_detections, color=(0, 255, 255)
    )

    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Draw Frame Number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(
            frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    save_vid(output_video_frames, "output_videos/output_video.mp4")


main()
