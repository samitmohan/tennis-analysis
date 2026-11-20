"""Court homography: exact round trips, and RANSAC outvoting a bad keypoint."""

import numpy as np

from court_lines.homography import (
    COURT_KPS_M, court_to_image, fit_homography, image_to_court, refine_keypoints,
)

# A plausible broadcast view: court metres -> 1280x720 pixels.
H_TRUE = np.array([[38.0, -9.0, 430.0], [0.0, 12.0, 150.0], [0.0, -0.012, 1.0]])


def test_court_dimensions():
    xs, ys = COURT_KPS_M[:, 0], COURT_KPS_M[:, 1]
    assert np.isclose(xs.max() - xs.min(), 10.97) and np.isclose(ys.max() - ys.min(), 23.77)
    # service lines sit 6.40 m either side of the net
    assert np.isclose(COURT_KPS_M[10, 1] - COURT_KPS_M[8, 1], 12.80)


def test_round_trip_is_exact():
    px = court_to_image(H_TRUE, COURT_KPS_M)
    back = image_to_court(H_TRUE, px)
    assert np.allclose(back, COURT_KPS_M, atol=1e-3)


def test_ransac_fixes_one_wild_keypoint():
    px = court_to_image(H_TRUE, COURT_KPS_M)
    rng = np.random.default_rng(0)
    noisy = px + rng.normal(0, 1.0, px.shape)
    noisy[3] += (60, -40)  # one keypoint 72 px off
    refined = refine_keypoints(noisy)
    err = np.linalg.norm(refined - px, axis=1)
    assert err[3] < 3.0 and err.max() < 3.0


def test_fit_fails_cleanly_on_garbage():
    rng = np.random.default_rng(1)
    assert fit_homography(rng.uniform(0, 1280, (14, 2))) is None
