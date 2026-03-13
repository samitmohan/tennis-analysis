from dataclasses import dataclass


@dataclass
class Rally:
    rally_id: int
    start_frame: int
    end_frame: int
    shot_count: int
    shot_frames: list[int]
    duration_seconds: float
    avg_shot_speed_kmh: float
    last_hitting_player: int | None


def detect_rallies(
    ball_shot_frames: list[int],
    fps: float,
    gap_threshold_seconds: float = 3.0,
    shot_speeds: dict[int, float] | None = None,
    shot_players: dict[int, int] | None = None,
) -> list[Rally]:
    """Group consecutive shots into rallies based on time gap between shots.

    A rally starts at the first shot and ends when the gap to the next shot
    exceeds gap_threshold_seconds.

    Args:
        ball_shot_frames: Frame indices where shots were detected.
        fps: Video frames per second.
        gap_threshold_seconds: Max gap (seconds) between shots in the same rally.
        shot_speeds: Optional mapping of shot frame -> ball speed in km/h.
        shot_players: Optional mapping of shot frame -> player id who hit the shot.
    """
    if not ball_shot_frames:
        return []

    gap_threshold_frames = gap_threshold_seconds * fps
    rallies: list[Rally] = []
    current_shots: list[int] = [ball_shot_frames[0]]

    for i in range(1, len(ball_shot_frames)):
        gap = ball_shot_frames[i] - ball_shot_frames[i - 1]
        if gap > gap_threshold_frames:
            rallies.append(
                _build_rally(len(rallies), current_shots, fps, shot_speeds, shot_players)
            )
            current_shots = [ball_shot_frames[i]]
        else:
            current_shots.append(ball_shot_frames[i])

    # Final rally
    rallies.append(
        _build_rally(len(rallies), current_shots, fps, shot_speeds, shot_players)
    )

    return rallies


def _build_rally(
    rally_id: int,
    shot_frames: list[int],
    fps: float,
    shot_speeds: dict[int, float] | None,
    shot_players: dict[int, int] | None,
) -> Rally:
    duration = (shot_frames[-1] - shot_frames[0]) / fps if len(shot_frames) > 1 else 0.0

    speeds = []
    if shot_speeds:
        for frame in shot_frames:
            if frame in shot_speeds:
                speeds.append(shot_speeds[frame])
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    last_player = None
    if shot_players and shot_frames[-1] in shot_players:
        last_player = shot_players[shot_frames[-1]]

    return Rally(
        rally_id=rally_id,
        start_frame=shot_frames[0],
        end_frame=shot_frames[-1],
        shot_count=len(shot_frames),
        shot_frames=shot_frames,
        duration_seconds=round(duration, 2),
        avg_shot_speed_kmh=round(avg_speed, 1),
        last_hitting_player=last_player,
    )
