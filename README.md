Project to learn YOLO

## Description
This project analyzes Tennis players in a video to measure their speed, ball shot speed and number of shots. This project will detect players and the tennis ball using YOLO and also utilizes CNNs to extract court keypoints. 

When you run the YOLO model on the video, it barely detects the ball (which we need for tennis analysis)
- Soln: Fine tune the yolo model on tennis ball
- ID Tracking is important, we need to detect which person is whom throughout the video/frames. Use yolo.track

TODO:
- Replace resnet50 with your own implementation
- Keypoints don't work: keypoints_model (Retrain model Google Collab)
- Add minicourt
- Add Player 1 Player 2 Stats
- Add DeepSort to player amd ball trackers