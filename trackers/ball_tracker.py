import cv2
import pickle
import pandas as pd
from ultralytics import YOLO
class BallTracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    # ball in the middle not getting detected
    def interpolate_ball_positions(self, ball_positions):
        ''' this ensures it draws frames for the missing object detections '''
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    # multiple frames
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections
        
        for frame in frames:
            player_dict = self.detect(frame)
            ball_detections.append(player_dict)
        # save this
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)
        return ball_detections



    def detect(self, frame): # image
        # only one ball so no track
        res = self.model.predict(frame, conf=0.15)[0] # save next frame ( can we use deepsort here ? )
        # k:pid, v:bounding box
        ball_detect = {}
        for box in res.boxes:
            result = box.xyxy.tolist()[0]
            ball_detect[1] = result
        return ball_detect
        
    def draw_boxes(self, video_frames, player_detections):
        output_vid_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # draw bounding box
            for track_id, boxx in ball_dict.items():
                x1, y1, x2, y2 = boxx
                cv2.putText(frame, f"Ball ID: {track_id}",(int(boxx[0]),int(boxx[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # draw this shit
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            output_vid_frames.append(frame)
        return output_vid_frames