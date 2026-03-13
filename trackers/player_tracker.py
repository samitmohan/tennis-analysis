import logging
import pickle

import cv2
from ultralytics import YOLO

from utils import measure_distance, get_center_of_bbox

logger = logging.getLogger(__name__)


class PlayerTracker:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def detect_frames(
        self,
        frames: list,
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> list[dict[int, list[float]]]:
        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)
        return player_detections

    def detect(self, frame) -> dict[int, list[float]]:
        res = self.model.track(frame, persist=True)[0]
        id_to_names = res.names
        player_dict = {}
        for box in res.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_to_names[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = result
        return player_dict

    def draw_boxes(self, video_frames: list, player_detections: list) -> list:
        output_vid_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, boxx in player_dict.items():
                x1, y1, x2, y2 = boxx
                cv2.putText(
                    frame, f"Player ID: {track_id}",
                    (int(boxx[0]), int(boxx[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                )
                frame = cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
            output_vid_frames.append(frame)
        return output_vid_frames

    def choose_and_filter_players(
        self,
        court_keypoints: list[float],
        player_detections: list[dict[int, list[float]]],
    ) -> list[dict[int, list[float]]]:
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {
                track_id: bbox
                for track_id, bbox in player_dict.items()
                if track_id in chosen_player
            }
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(
        self,
        court_keypoints: list[float],
        player_dict: dict[int, list[float]],
    ) -> list[int]:
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        distances.sort(key=lambda x: x[1])

        if len(distances) < 2:
            logger.warning(
                "Fewer than 2 players detected (%d found). "
                "Returning all detected players.",
                len(distances),
            )
            return [d[0] for d in distances]

        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players
