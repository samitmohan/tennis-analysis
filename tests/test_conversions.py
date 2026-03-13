import pytest

from utils.conversions import (
    convert_pixel_distance_to_meters,
    convert_meters_to_pixel_distance,
)


class TestPixelToMeters:
    def test_identity(self):
        # When reference matches 1:1
        result = convert_pixel_distance_to_meters(100, 100.0, 100.0)
        assert result == 100.0

    def test_scaling(self):
        # 50 pixels, reference is 10m = 200px -> 50 * 10 / 200 = 2.5m
        result = convert_pixel_distance_to_meters(50, 10.0, 200.0)
        assert abs(result - 2.5) < 1e-9

    def test_zero_distance(self):
        result = convert_pixel_distance_to_meters(0, 10.0, 100.0)
        assert result == 0.0


class TestMetersToPixels:
    def test_identity(self):
        result = convert_meters_to_pixel_distance(100, 100.0, 100.0)
        assert result == 100.0

    def test_scaling(self):
        # 2.5m, reference is 10m = 200px -> 2.5 * 200 / 10 = 50px
        result = convert_meters_to_pixel_distance(2.5, 10.0, 200.0)
        assert abs(result - 50.0) < 1e-9

    def test_round_trip(self):
        # Converting to meters and back should give original pixel distance
        original_px = 75.0
        ref_m = 10.97
        ref_px = 210.0
        meters = convert_pixel_distance_to_meters(original_px, ref_m, ref_px)
        recovered_px = convert_meters_to_pixel_distance(meters, ref_m, ref_px)
        assert abs(recovered_px - original_px) < 1e-9
