from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    player_detector: str = "yolov8x"
    ball_detector: str = "models/last.pt"
    court_keypoint: str = "models/keypointsModel.pth"
    ball_confidence: float = 0.15


@dataclass
class CourtConfig:
    single_line_width: float = 8.23
    double_line_width: float = 10.97
    half_court_line_height: float = 11.88
    service_line_width: float = 6.4
    double_alley_difference: float = 1.37
    no_mans_land_height: float = 5.48


@dataclass
class PlayerConfig:
    player_1_height_meters: float = 1.88
    player_2_height_meters: float = 1.91


@dataclass
class ShotDetectionConfig:
    rolling_window: int = 5
    minimum_change_frames: int = 25


@dataclass
class MiniCourtConfig:
    width: int = 250
    height: int = 500
    buffer: int = 50
    padding: int = 20


@dataclass
class RallyConfig:
    gap_threshold_seconds: float = 3.0


@dataclass
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    players: PlayerConfig = field(default_factory=PlayerConfig)
    shot_detection: ShotDetectionConfig = field(default_factory=ShotDetectionConfig)
    mini_court: MiniCourtConfig = field(default_factory=MiniCourtConfig)
    rally: RallyConfig = field(default_factory=RallyConfig)


def _merge_dict_into_dataclass(dc: object, data: dict) -> None:
    """Recursively merge a dict into a dataclass instance."""
    for key, value in data.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, '__dataclass_fields__'):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(dc, key, value)


def load_config(config_path: Path | str | None = None) -> Config:
    """Load config from YAML file, falling back to defaults."""
    config = Config()
    if config_path is None:
        default_path = Path(__file__).parent / "config" / "default.yaml"
        if default_path.exists():
            config_path = default_path

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            _merge_dict_into_dataclass(config, data)

    return config
