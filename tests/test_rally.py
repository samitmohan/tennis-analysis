from analytics.rally import detect_rallies


class TestDetectRallies:
    def test_single_rally(self):
        """All shots within gap threshold form one rally."""
        # Shots every 30 frames at 30 FPS = 1 second apart, threshold is 3 seconds
        shot_frames = [30, 60, 90, 120, 150]
        rallies = detect_rallies(shot_frames, fps=30.0, gap_threshold_seconds=3.0)
        assert len(rallies) == 1
        assert rallies[0].shot_count == 5
        assert rallies[0].start_frame == 30
        assert rallies[0].end_frame == 150

    def test_two_rallies(self):
        """Gap larger than threshold splits into two rallies."""
        # Rally 1: frames 30, 60, 90 (1s apart)
        # Gap: 90 -> 300 = 7 seconds (> 3s threshold)
        # Rally 2: frames 300, 330, 360
        shot_frames = [30, 60, 90, 300, 330, 360]
        rallies = detect_rallies(shot_frames, fps=30.0, gap_threshold_seconds=3.0)
        assert len(rallies) == 2
        assert rallies[0].shot_count == 3
        assert rallies[1].shot_count == 3

    def test_empty_shots(self):
        """No shots means no rallies."""
        rallies = detect_rallies([], fps=30.0)
        assert rallies == []

    def test_single_shot(self):
        """One shot forms one rally with count 1."""
        rallies = detect_rallies([100], fps=30.0)
        assert len(rallies) == 1
        assert rallies[0].shot_count == 1
        assert rallies[0].duration_seconds == 0.0

    def test_rally_duration(self):
        """Rally duration should be (last_frame - first_frame) / fps."""
        shot_frames = [0, 30, 60]
        rallies = detect_rallies(shot_frames, fps=30.0)
        assert rallies[0].duration_seconds == 2.0

    def test_with_shot_speeds(self):
        """Shot speeds are averaged per rally."""
        shot_frames = [10, 40, 70]
        shot_speeds = {10: 100.0, 40: 120.0, 70: 110.0}
        rallies = detect_rallies(
            shot_frames, fps=30.0, shot_speeds=shot_speeds
        )
        assert rallies[0].avg_shot_speed_kmh == 110.0

    def test_with_shot_players(self):
        """Last hitting player is recorded."""
        shot_frames = [10, 40, 70]
        shot_players = {10: 1, 40: 2, 70: 1}
        rallies = detect_rallies(
            shot_frames, fps=30.0, shot_players=shot_players
        )
        assert rallies[0].last_hitting_player == 1

    def test_gap_at_exact_threshold(self):
        """Gap exactly at threshold should stay in the same rally."""
        # 90 frames at 30 FPS = exactly 3.0 seconds
        shot_frames = [0, 90]
        rallies = detect_rallies(shot_frames, fps=30.0, gap_threshold_seconds=3.0)
        assert len(rallies) == 1

    def test_gap_just_over_threshold(self):
        """Gap just over threshold should split rallies."""
        # 91 frames at 30 FPS = 3.03 seconds > 3.0 threshold
        shot_frames = [0, 91]
        rallies = detect_rallies(shot_frames, fps=30.0, gap_threshold_seconds=3.0)
        assert len(rallies) == 2
