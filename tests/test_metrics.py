"""The benchmark rules, checked case by case against the reference definitions."""

import pytest

from evaluation.metrics import (
    add_counts, match_events, prf, score_ball, score_court, summarize_ball,
)


def test_ball_rule_covers_all_four_cases():
    preds = [(100, 100), (100, 100), (100, 100), None, None]
    gts = [(103, 104), (110, 100), None, (50, 50), None]
    c = score_ball(preds, gts)
    # 5 px away exactly is not < 5; (103,104) is 5.0 away
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (0, 3, 1, 1)


def test_ball_tp_and_median():
    c = score_ball([(0, 0), (0, 0), (0, 0)], [(3, 0), (0, 1), (4, 0)])
    s = summarize_ball(c)
    assert s["tp"] == 3 and s["median_px"] == 3
    assert s["precision"] == 1.0 and s["recall"] == 1.0


def test_ball_counts_add_across_clips():
    a = score_ball([(0, 0)], [(1, 0)])
    b = score_ball([None], [(1, 0)])
    s = summarize_ball(add_counts(a, b))
    assert (s["tp"], s["fn"]) == (1, 1) and s["recall"] == 0.5


def test_ball_length_mismatch_fails_loud():
    with pytest.raises(ValueError):
        score_ball([None], [None, None])


def test_events_one_to_one_within_tolerance():
    # pred 10 matches gt 12; pred 11 cannot reuse gt 12; gt 40 has nothing within 5
    assert match_events([10, 11, 30], [12, 40], tol=5) == {"tp": 1, "fp": 2, "fn": 1}


def test_events_closest_pair_wins():
    # gt 20 is closest to pred 21, so pred 16 must take gt 12 (distance 4)
    assert match_events([16, 21], [12, 20], tol=5) == {"tp": 2, "fp": 0, "fn": 0}


def test_prf_empty_is_zero():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_court_rule():
    preds = [(10, 10), (10, 10), (2000, 10), (10, 10), None]
    gts = [(15, 10), (20, 10), (10, 10), None, None]
    s = score_court(preds, gts)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 2, 1, 1)
    assert s["median_px"] == 7.5  # distances 5 and 10
