from utils import read_vid, save_vid
import cv2
from trackers import PlayerTracker, BallTracker
from court_lines import CourtLineDetect

def main():
    inp_vid_path = 'input/input_video.mp4'
    video_frames = read_vid(inp_vid_path)
    # detect players
    player_tracker = PlayerTracker('yolov8x')
    # detect balls
    ball_tracker = BallTracker('models/last.pt')

    # draw bounding box on top of frames
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # court line detector model
    court_model_path = 'models/keypointsModel.pth'
    court_line_detect = CourtLineDetect(court_model_path)
    court_keypoints = court_line_detect.predict(video_frames[0])

    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # draw output
    output_video_frames = player_tracker.draw_boxes(video_frames, player_detections) # player draw
    output_video_frames = ball_tracker.draw_boxes(output_video_frames, ball_detections) # ball draw

    # Ensure keypoints are drawn last
    output_video_frames = court_line_detect.draw_keypts_vid(output_video_frames, court_keypoints) # key points draw

    # Draw Frame Number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_vid(output_video_frames, 'output_videos/output_video.mp4')


main()