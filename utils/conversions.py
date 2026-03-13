def convert_pixel_distance_to_meters(
    pixel_distance: float,
    reference_height_in_meters: float,
    reference_height_in_pixels: float,
) -> float:
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels


def convert_meters_to_pixel_distance(
    meters: float,
    reference_height_in_meters: float,
    reference_height_in_pixels: float,
) -> float:
    return (meters * reference_height_in_pixels) / reference_height_in_meters