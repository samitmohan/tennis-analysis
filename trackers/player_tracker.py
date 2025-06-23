import cv2
import pickle
import math
from ultralytics import YOLO
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    # multiple frames
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect(frame)
            player_detections.append(player_dict)
        # save this
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)
        return player_detections

    def detect(self, frame): # image
        res = self.model.track(frame, persist=True)[0] # save next frame ( can we use deepsort here ? )
        id_to_names = res.names
        # k:pid, v:bounding box
        player_dict = {}
        # only include people
        for box in res.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_to_names[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = result
        return player_dict
        
    def draw_boxes(self, video_frames, player_detections):
        output_vid_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # draw bounding box
            for track_id, boxx in player_dict.items():
                x1, y1, x2, y2 = boxx
                cv2.putText(frame, f"Player ID: {track_id}",(int(boxx[0]),int(boxx[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # draw this shit
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            output_vid_frames.append(frame)
        return output_vid_frames


    # choose players (proximity to their court) : we don't want to see all people, only the players (closest to the keypoints)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sort the distances in ascending order
        distances.sort(key=lambda x: x[1])
        
        # Choose the first 2 tracks, or all tracks if less than 2
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players
