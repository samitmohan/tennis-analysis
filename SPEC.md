# SPEC: make tennis-analysis measurable, then better

This project started from [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis)
(YOLOv8 players, a fine-tuned YOLO ball detector, a ResNet50 court-keypoint regressor,
a mini court scaled by player height). The goal is to measure that pipeline against
labelled data, fix what the numbers show is broken, and replace the weakest parts.

## Done means

`results/` holds JSON from one command per benchmark, and the README shows a
before/after table built from those files. Every number in the README comes from a
file in `results/`. No number is typed by hand.

## Benchmarks

| What | Data | Metric | Matches |
| --- | --- | --- | --- |
| Ball position | TrackNet dataset (10 broadcast games, 1280x720, 30 fps). Train on games 1-7, pick checkpoints on game 8, test on games 9-10 once. Split by game so no clip leaks into test. | Per frame: TP if prediction within 5 px of the label; FP if too far or the ball is not visible; FN if no prediction for a visible ball; TN otherwise. Precision, recall, F1, median px error over TPs. | yastrebksv/TrackNet `validate(min_dist=5)` |
| Hits and bounces | TrackNet labels: status 1 = hit, 2 = bounce, test games | Event P/R/F1, one-to-one matching within +-5 frames (167 ms) | ours |
| Court keypoints | TennisCourtDetector val split (25% of 8,841 images, 1280x720, images with all 14 points inside the frame) | Per keypoint TP if < 7 px. Precision = TP/(TP+FP), accuracy = (TP+TN)/all, median px distance | yastrebksv/TennisCourtDetector `base_validator.py` (published base model: 0.936 / 0.933 / 2.83 px) |
| Throughput | `input_videos/input_video.mp4` and a 10-minute video | End-to-end frames per second, peak RSS | ours |
| Serve speed | Broadcast clips with the on-screen radar speed | Mean absolute error in km/h against the radar | blocked: needs clips |

## Phases

1. **Baseline.** Run the benchmarks on the tutorial pipeline exactly as it was
   (tutorial weights, last-box ball bug included). Save to `results/baseline_*.json`.
2. **Fixes.** Highest-confidence ball box; stream frames instead of holding the whole
   video in RAM; detect the court per scene, not once at frame 0. Re-run.
3. **Temporal ball tracker.** A TrackNet-style network (3 stacked frames in, a heatmap out),
   trained here on games 1-8. Re-run ball and event benchmarks.
4. **Geometry.** Homography from the 14 keypoints (RANSAC). Project only ground contacts:
   player feet and ball bounces. Shot speed = hit-to-bounce distance on the court / time.
   In/out call per bounce.
5. **Real-time.** Batched, fp16 inference; TensorRT on the RTX 3090. Report fps and peak RAM.
6. **README.** Credit the tutorial, list what changed, show the table and a failure gallery.

## Machines

Editing and CPU work on the MacBook. Training and GPU evaluation on bqa3 (RTX 3090).
bqa3's root disk is shared and nearly full: keep datasets unzipped once, delete zips,
and check `df -h ~` before staging anything.
