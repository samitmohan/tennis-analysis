"""Heatmap decoding and sample building, checked without a trained model."""

import numpy as np
import torch

from ball_tracking.data import IN_H, IN_W, decode, gaussian, window
from ball_tracking.model import TrackNet, weighted_bce


def test_decode_recovers_the_gaussian_centre_in_frame_pixels():
    prob = gaussian(100.0, 50.0, 2.0)
    x, y = decode(prob, 1280, 720)
    assert abs(x - 100 * 1280 / IN_W) < 0.5 and abs(y - 50 * 720 / IN_H) < 0.5


def test_decode_ignores_a_second_weaker_blob():
    prob = gaussian(100.0, 50.0, 2.0) + 0.8 * gaussian(400.0, 200.0, 2.0)
    x, _ = decode(prob, IN_W, IN_H)
    assert abs(x - 100) < 0.5


def test_decode_returns_none_below_threshold():
    assert decode(np.full((IN_H, IN_W), 0.3, np.float32), 1280, 720) is None


def test_window_repeats_first_frame():
    assert window(0) == [0, 0, 0] and window(1) == [0, 0, 1] and window(5) == [3, 4, 5]


def test_model_output_shape_and_loss_is_finite():
    model = TrackNet().eval()
    x = torch.rand(1, 9, IN_H, IN_W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, IN_H, IN_W)
    target = torch.from_numpy(gaussian(10.0, 10.0, 2.0))[None, None]
    assert torch.isfinite(weighted_bce(out, target))


def test_stream_predict_matches_window_semantics_and_batching():
    """Streaming over frames must equal stacking windows by hand, for any batch size."""
    from ball_tracking.data import stack_frames, to_input
    from ball_tracking.infer import stream_predict

    torch.manual_seed(0)
    model = TrackNet().eval()
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (72, 128, 3), dtype=np.uint8) for _ in range(5)]
    with torch.no_grad():
        ref = []
        for i in range(5):
            x = stack_frames([frames[j] for j in window(i)])[None]
            ref.append(torch.sigmoid(model(to_input(torch.from_numpy(x), "cpu"))).numpy()[0, 0])
    got_b2 = list(stream_predict(model, frames, "cpu", batch=2, thr=0.0))
    got_b16 = list(stream_predict(model, frames, "cpu", batch=16, thr=0.0))
    want = [decode(p, 128, 72, thr=0.0) for p in ref]
    assert got_b2 == got_b16
    assert all(np.allclose(g, w, atol=1e-3) for g, w in zip(got_b2, want))
