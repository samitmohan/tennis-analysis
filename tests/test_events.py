"""Trajectory features and event picking, without a trained model."""

import numpy as np

from ball_tracking.events import W, features, fill_gaps, pick


def test_fill_gaps_interpolates_short_gaps_only():
    pts = [(0.0, 0.0), None, None, (30.0, 0.0)] + [None] * 10 + [(100.0, 0.0)]
    xy = fill_gaps(pts, max_gap=6)
    assert np.allclose(xy[1], (10, 0)) and np.allclose(xy[2], (20, 0))
    assert np.isnan(xy[5, 0])  # a 10-frame gap stays empty


def test_features_shape_and_relative_positions():
    pts = [(float(i), 2.0 * i) for i in range(20)]
    f = features(pts)
    assert f.shape == (20, 2 * (2 * W + 1) + (2 * W + 1) + 1)
    assert features(pts, own_y=True).shape[1] == f.shape[1] + 1
    # frame 10, offset +1: dx = 1 px, dy = 2 px, in units of 100 px
    rel = f[10, : 2 * (2 * W + 1)].reshape(-1, 2)
    assert np.allclose(rel[W + 1], (0.01, 0.02)) and np.allclose(rel[W], (0, 0))


def test_pick_keeps_peaks_apart():
    p = np.zeros(40)
    p[10], p[12], p[30] = 0.9, 0.8, 0.7
    assert pick(p, thr=0.5, nms=5) == [10, 30]
    assert pick(p, thr=0.95) == []


def test_mirror_flips_dx_only():
    from ball_tracking.events import mirror

    pts = [(float(3 * i), 2.0 * i) for i in range(20)]
    f = features(pts)
    m = mirror(f)
    rel, mrel = f[10, : 2 * (2 * W + 1)].reshape(-1, 2), m[10, : 2 * (2 * W + 1)].reshape(-1, 2)
    assert np.allclose(mrel[:, 0], -rel[:, 0]) and np.allclose(mrel[:, 1], rel[:, 1])
    assert np.allclose(m[:, 2 * (2 * W + 1):], f[:, 2 * (2 * W + 1):])
