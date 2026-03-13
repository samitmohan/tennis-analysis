from utils.box_utils import (
    get_center_of_bbox,
    get_foot_position,
    get_height_of_bbox,
    measure_distance,
    measure_xy_distance,
)


class TestGetCenterOfBbox:
    def test_basic(self):
        assert get_center_of_bbox([0, 0, 100, 100]) == (50, 50)

    def test_offset(self):
        assert get_center_of_bbox([10, 20, 30, 40]) == (20, 30)

    def test_float_coords(self):
        center = get_center_of_bbox([0.0, 0.0, 99.0, 99.0])
        assert center == (49, 49)

    def test_non_square(self):
        assert get_center_of_bbox([0, 0, 200, 100]) == (100, 50)


class TestMeasureDistance:
    def test_zero_distance(self):
        assert measure_distance((0, 0), (0, 0)) == 0.0

    def test_horizontal(self):
        assert measure_distance((0, 0), (3, 0)) == 3.0

    def test_vertical(self):
        assert measure_distance((0, 0), (0, 4)) == 4.0

    def test_diagonal(self):
        # 3-4-5 triangle
        assert measure_distance((0, 0), (3, 4)) == 5.0

    def test_negative_coords(self):
        dist = measure_distance((-1, -1), (2, 3))
        assert abs(dist - 5.0) < 1e-9


class TestGetFootPosition:
    def test_basic(self):
        assert get_foot_position([10, 20, 30, 80]) == (20, 80)

    def test_returns_bottom_center(self):
        x, y = get_foot_position([0, 0, 100, 200])
        assert x == 50
        assert y == 200


class TestGetHeightOfBbox:
    def test_basic(self):
        assert get_height_of_bbox([0, 10, 50, 90]) == 80

    def test_zero_height(self):
        assert get_height_of_bbox([0, 50, 100, 50]) == 0


class TestMeasureXYDistance:
    def test_basic(self):
        dx, dy = measure_xy_distance((10, 20), (30, 50))
        assert dx == 20
        assert dy == 30

    def test_same_point(self):
        dx, dy = measure_xy_distance((5, 5), (5, 5))
        assert dx == 0
        assert dy == 0

    def test_always_positive(self):
        dx, dy = measure_xy_distance((30, 50), (10, 20))
        assert dx == 20
        assert dy == 30
