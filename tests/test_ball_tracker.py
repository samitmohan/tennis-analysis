"""Test ball shot detection with synthetic position data (no model required)."""

from trackers.ball_tracker import BallTracker


def _make_ball_positions(y_values: list[float]) -> list[dict[int, list[float]]]:
    """Create synthetic ball position dicts from a list of Y-coordinates.

    Each position has a fixed X and the given Y, with a small bbox around it.
    """
    positions = []
    for y in y_values:
        positions.append({1: [100.0, y - 5, 110.0, y + 5]})
    return positions


class TestGetBallShotFrames:
    def test_v_pattern_single_hit(self):
        """Ball goes down then up - one direction change = one hit."""
        # Going down (y increasing) then up (y decreasing)
        y_vals = (
            [200 + i * 3 for i in range(40)]  # going down
            + [200 + 40 * 3 - i * 3 for i in range(40)]  # going up
            + [200 - i * 3 for i in range(40)]  # continuing up
        )
        positions = _make_ball_positions(y_vals)
        tracker = BallTracker.__new__(BallTracker)
        frames = tracker.get_ball_shot_frames(
            positions, rolling_window=5, minimum_change_frames=25
        )
        assert len(frames) >= 1

    def test_no_movement_no_hits(self):
        """Constant Y position - no direction changes."""
        y_vals = [200.0] * 100
        positions = _make_ball_positions(y_vals)
        tracker = BallTracker.__new__(BallTracker)
        frames = tracker.get_ball_shot_frames(
            positions, rolling_window=5, minimum_change_frames=25
        )
        assert len(frames) == 0

    def test_empty_positions(self):
        """Empty input should return empty list."""
        tracker = BallTracker.__new__(BallTracker)
        frames = tracker.get_ball_shot_frames(
            [], rolling_window=5, minimum_change_frames=25
        )
        assert frames == []

    def test_monotonic_no_hits(self):
        """Ball moving in one direction only - no hits."""
        y_vals = [100 + i * 2 for i in range(100)]
        positions = _make_ball_positions(y_vals)
        tracker = BallTracker.__new__(BallTracker)
        frames = tracker.get_ball_shot_frames(
            positions, rolling_window=5, minimum_change_frames=25
        )
        assert len(frames) == 0
