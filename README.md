# Tennis Match Analysis

Computer vision pipeline that analyzes tennis match video to track players and ball, compute shot and movement speeds, detect rallies, and generate position heatmaps.

## Demo

![Output](output_videos/output.gif)

## Features

- **Player detection and tracking** via YOLOv8x with persistent ID assignment
- **Ball detection** using a fine-tuned YOLO model trained on tennis ball data
- **Court keypoint detection** via ResNet50 CNN (14 keypoints)
- **Coordinate transformation** from camera view to bird's-eye mini court
- **Shot speed and player speed** computation per shot event
- **Rally detection** - groups shots into rallies with per-rally statistics
- **Player position heatmaps** - Gaussian-smoothed court heatmaps per player
- **Statistics export** to JSON and CSV
- **Configurable** via YAML config with CLI overrides

## Architecture

```
Input Video
    |
    v
+-------------------+    +------------------+    +---------------------+
| Player Tracker    |    | Ball Tracker     |    | Court Keypoint      |
| (YOLOv8x)        |    | (Fine-tuned YOLO)|    | Detector (ResNet50) |
+-------------------+    +------------------+    +---------------------+
    |                         |                          |
    v                         v                          v
+----------------------------------------------------------------+
|              Coordinate Transform (Mini Court)                  |
+----------------------------------------------------------------+
    |                    |                       |
    v                    v                       v
+----------------+  +-----------------+  +------------------+
| Shot Detection |  | Rally Detection |  | Speed Calculation|
+----------------+  +-----------------+  +------------------+
    |                    |                       |
    v                    v                       v
+----------------------------------------------------------------+
|                      Output Generation                          |
|  Annotated Video  |  JSON/CSV Stats  |  Heatmap PNGs           |
+----------------------------------------------------------------+
```

## Installation

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/tennis-analysis.git
cd tennis-analysis

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Model Weights

Download the pre-trained model weights and place them in the `models/` directory:

| Model | File | Purpose |
|-------|------|---------|
| YOLOv8x | Downloaded automatically | Player detection |
| Ball detector | `models/last.pt` | Fine-tuned ball detection |
| Court keypoints | `models/keypointsModel.pth` | Court line keypoint regression |

## Usage

```bash
# Basic analysis with output video
tennis-analyze -i input/input_video.mp4 -o output/

# Export statistics without generating video
tennis-analyze -i input/input_video.mp4 --export-stats stats.json --no-video

# Generate player heatmaps
tennis-analyze -i input/input_video.mp4 -o output/ --heatmap

# Use cached detection stubs (skip model inference)
tennis-analyze -i input/input_video.mp4 -o output/ --use-stubs

# Custom config with verbose logging
tennis-analyze -i input/input_video.mp4 -o output/ -c config/default.yaml -v

# Full analysis pipeline
tennis-analyze -i input/input_video.mp4 -o output/ --export-stats output/stats.json --heatmap -v
```

### Configuration

All parameters are configurable via `config/default.yaml`:

```yaml
models:
  player_detector: "yolov8x"
  ball_detector: "models/last.pt"
  ball_confidence: 0.15

court:
  double_line_width: 10.97  # meters

players:
  player_1_height_meters: 1.88
  player_2_height_meters: 1.91

rally:
  gap_threshold_seconds: 3.0
```

### JSON Output Example

```json
{
  "video": {
    "fps": 24.0,
    "total_frames": 382,
    "duration_seconds": 15.92
  },
  "player_summary": {
    "player_1": {
      "total_shots": 4,
      "avg_shot_speed_kmh": 72.3
    }
  },
  "rallies": [
    {
      "rally_id": 0,
      "shot_count": 7,
      "duration_seconds": 12.5,
      "avg_shot_speed_kmh": 68.4
    }
  ]
}
```

## How It Works

1. **Player detection**: YOLOv8x detects all people in each frame. The two players closest to court keypoints are selected and tracked across frames.

2. **Ball detection**: A YOLO model fine-tuned on tennis ball data detects the ball. Pandas interpolation fills detection gaps. Shot events are identified by detecting direction changes in the ball's vertical trajectory.

3. **Court detection**: A ResNet50 model predicts 14 court keypoints from the first frame. These keypoints establish the mapping between camera coordinates and real-world court coordinates.

4. **Coordinate transform**: Player foot positions and ball positions are projected onto a mini court using the court keypoints as reference. Player heights provide the pixel-to-meter scale factor.

5. **Speed computation**: Ball speed is calculated from mini court distance between consecutive shot frames. Player speed tracks opponent movement during each shot interval.

6. **Rally detection**: Consecutive shots are grouped into rallies. A new rally starts when the gap between shots exceeds a configurable threshold (default 3 seconds).

7. **Heatmaps**: Player positions are accumulated into a 2D histogram on the mini court, Gaussian-blurred, and rendered with a JET colormap.

## Project Structure

```
tennis-analysis/
  cli.py                  # CLI entry point (argparse)
  pipeline.py             # Main analysis pipeline orchestrator
  config.py               # Dataclass-based YAML config loader
  main.py                 # Legacy entry point
  config/
    default.yaml          # Default configuration values
  trackers/
    player_tracker.py     # YOLOv8x player detection and tracking
    ball_tracker.py       # Ball detection, interpolation, shot detection
  court_lines/
    court_line_detect.py  # ResNet50 court keypoint prediction
  mini_court/
    mini_court.py         # Mini court visualization and coordinate transform
  analytics/
    rally.py              # Rally detection from shot events
    export.py             # JSON/CSV statistics export
    heatmap.py            # Player position heatmap generation
  utils/
    video_utils.py        # Video I/O with FPS handling
    box_utils.py          # Bounding box geometry utilities
    conversions.py        # Pixel-to-meter coordinate conversions
    player_stats.py       # On-frame statistics overlay
  constants/
    __init__.py           # Court dimension constants
  tests/
    test_box_utils.py     # Bounding box utility tests
    test_conversions.py   # Coordinate conversion tests
    test_ball_tracker.py  # Shot detection tests (synthetic data)
    test_rally.py         # Rally grouping tests
  models/                 # Model weight files (not tracked in git)
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Limitations and Future Work

- **Single camera angle**: Assumes a fixed broadcast-style camera. Pan/zoom or player-mounted cameras would require re-calibration per frame.
- **Two-player singles only**: The player selection heuristic picks the two closest people to court keypoints. Doubles would need a different approach.
- **Static court detection**: Court keypoints are predicted from the first frame only. Camera movement would require per-frame or periodic re-detection.
- **No shot type classification**: Distinguishing forehands, backhands, and serves would require pose estimation and labeled training data.
- **No score tracking**: Automated scoring from broadcast video would need OCR on the scoreboard overlay, which varies by broadcaster.
