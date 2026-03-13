"""CLI entry point for tennis-analyze."""

import argparse
import logging
import sys

from config import load_config
from pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tennis-analyze",
        description="Analyze tennis match video: detect players, track ball, "
        "compute shot/player speeds, detect rallies, generate heatmaps.",
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "-o", "--output", default="output_videos",
        help="Output directory (default: output_videos)",
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to custom YAML config file",
    )
    parser.add_argument(
        "--models-dir",
        help="Directory containing model weight files",
    )
    parser.add_argument(
        "--export-stats",
        help="Path to export match statistics as JSON",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip output video generation (stats/heatmaps only)",
    )
    parser.add_argument(
        "--heatmap", action="store_true",
        help="Generate player position heatmap PNGs",
    )
    parser.add_argument(
        "--use-stubs", action="store_true",
        help="Use cached detection stubs instead of running models",
    )
    parser.add_argument(
        "--stubs-dir", default="tracker_stubs",
        help="Directory containing detection stub files (default: tracker_stubs)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging output",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    # Override model paths if models-dir is specified
    if args.models_dir:
        from pathlib import Path
        models_dir = Path(args.models_dir)
        if (models_dir / "last.pt").exists():
            config.models.ball_detector = str(models_dir / "last.pt")
        if (models_dir / "keypointsModel.pth").exists():
            config.models.court_keypoint = str(models_dir / "keypointsModel.pth")

    results = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        config=config,
        use_stubs=args.use_stubs,
        stubs_dir=args.stubs_dir,
        export_stats=args.export_stats,
        generate_heatmaps=args.heatmap,
        no_video=args.no_video,
    )

    # Print summary
    print(f"\nAnalysis complete:")
    print(f"  Frames: {results['total_frames']}")
    print(f"  FPS: {results['fps']:.1f}")
    print(f"  Shots detected: {len(results['ball_shot_frames'])}")
    print(f"  Rallies: {len(results['rallies'])}")

    for rally in results["rallies"]:
        print(
            f"    Rally {rally.rally_id + 1}: "
            f"{rally.shot_count} shots, "
            f"{rally.duration_seconds:.1f}s"
        )


if __name__ == "__main__":
    main()
