import logging
import pickle

import cv2
import pandas as pd
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class BallTracker:
    def __init__(self, model_path: str, confidence: float = 0.15) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence

    def interpolate_ball_positions(
        self, ball_positions: list[dict[int, list[float]]]
    ) -> list[dict[int, list[float]]]:
        """Fill gaps in ball detection using pandas interpolation."""
        ball_positions_list = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions_list, columns=['x1', 'y1', 'x2', 'y2']
        )
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        return [{1: x} for x in df_ball_positions.to_numpy().tolist()]

    def get_ball_shot_frames(
        self,
        ball_positions: list[dict[int, list[float]]],
        rolling_window: int = 5,
        minimum_change_frames: int = 25,
    ) -> list[int]:
        """Detect frames where ball direction changes (shot events)."""
        ball_positions_list = [x.get(1, []) for x in ball_positions]
        df = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])

        df['ball_hit'] = 0
        df['mid_y'] = (df['y1'] + df['y2']) / 2
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(
            window=rolling_window, min_periods=1, center=False
        ).mean()
        df['delta_y'] = df['mid_y_rolling_mean'].diff()

        for i in range(1, len(df) - int(minimum_change_frames * 1.2)):
            negative_change = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i + 1] < 0
            positive_change = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i + 1] > 0

            if negative_change or positive_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames * 1.2) + 1):
                    neg_following = (
                        df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[change_frame] < 0
                    )
                    pos_following = (
                        df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[change_frame] > 0
                    )
                    if negative_change and neg_following:
                        change_count += 1
                    elif positive_change and pos_following:
                        change_count += 1

                if change_count > minimum_change_frames - 1:
                    df.loc[i, 'ball_hit'] = 1

        return df[df['ball_hit'] == 1].index.tolist()

    def detect_frames(
        self,
        frames: list,
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> list[dict[int, list[float]]]:
        ball_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)
        return ball_detections

    def detect(self, frame) -> dict[int, list[float]]:
        res = self.model.predict(frame, conf=self.confidence)[0]
        ball_detect = {}
        for box in res.boxes:
            result = box.xyxy.tolist()[0]
            ball_detect[1] = result
        return ball_detect

    def draw_boxes(self, video_frames: list, ball_detections: list) -> list:
        output_vid_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, boxx in ball_dict.items():
                x1, y1, x2, y2 = boxx
                cv2.putText(
                    frame, f"Ball ID: {track_id}",
                    (int(boxx[0]), int(boxx[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )
                frame = cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                )
            output_vid_frames.append(frame)
        return output_vid_frames
