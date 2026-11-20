"""The flip permutation must mirror the real court, or augmentation teaches wrong labels."""

import numpy as np

from court_lines.heatmap import FLIP
from court_lines.homography import COURT_KPS_M, DOUBLES_W


def test_flip_is_an_involution():
    assert [FLIP[FLIP[i]] for i in range(14)] == list(range(14))


def test_flip_matches_mirrored_court_geometry():
    mirrored = COURT_KPS_M.copy()
    mirrored[:, 0] = DOUBLES_W - mirrored[:, 0]
    # after a flip, keypoint i sits where old keypoint FLIP[i] was, mirrored
    assert np.allclose(mirrored[FLIP], COURT_KPS_M, atol=1e-5)
