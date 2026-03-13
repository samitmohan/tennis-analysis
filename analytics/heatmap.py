import cv2
import numpy as np


def generate_heatmap(
    player_positions: list[dict[int, tuple[int, int]]],
    court_bounds: tuple[int, int, int, int],
    output_path: str,
    player_id: int = 1,
    resolution: tuple[int, int] = (400, 800),
    blur_sigma: int = 20,
) -> np.ndarray:
    """Generate a bird's-eye court heatmap from player positions.

    Args:
        player_positions: Per-frame dict mapping player_id -> (x, y) mini court coords.
        court_bounds: (start_x, start_y, end_x, end_y) of the mini court area.
        output_path: Path to save the heatmap PNG.
        player_id: Which player to generate the heatmap for.
        resolution: Output image (width, height).
        blur_sigma: Gaussian blur kernel size for smoothing.
    """
    start_x, start_y, end_x, end_y = court_bounds
    court_w = end_x - start_x
    court_h = end_y - start_y

    # Accumulate positions into a 2D histogram
    hist = np.zeros((resolution[1], resolution[0]), dtype=np.float32)

    for frame_positions in player_positions:
        if player_id not in frame_positions:
            continue
        px, py = frame_positions[player_id]
        # Normalize to court bounds
        norm_x = (px - start_x) / court_w
        norm_y = (py - start_y) / court_h

        # Map to histogram resolution
        hx = int(np.clip(norm_x * resolution[0], 0, resolution[0] - 1))
        hy = int(np.clip(norm_y * resolution[1], 0, resolution[1] - 1))
        hist[hy, hx] += 1

    # Gaussian blur for smooth heatmap
    kernel_size = blur_sigma * 2 + 1
    hist = cv2.GaussianBlur(hist, (kernel_size, kernel_size), blur_sigma)

    # Normalize to 0-255
    if hist.max() > 0:
        hist = (hist / hist.max() * 255).astype(np.uint8)
    else:
        hist = hist.astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(hist, cv2.COLORMAP_JET)

    # Draw court lines on top
    heatmap_colored = _draw_court_overlay(heatmap_colored, resolution)

    cv2.imwrite(output_path, heatmap_colored)
    return heatmap_colored


def _draw_court_overlay(
    image: np.ndarray, resolution: tuple[int, int]
) -> np.ndarray:
    """Draw simplified court lines on the heatmap."""
    w, h = resolution
    color = (255, 255, 255)
    thickness = 1

    # Outer court rectangle
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)
    cv2.rectangle(
        image,
        (margin_x, margin_y),
        (w - margin_x, h - margin_y),
        color, thickness,
    )

    # Net line (horizontal center)
    net_y = h // 2
    cv2.line(image, (margin_x, net_y), (w - margin_x, net_y), color, thickness + 1)

    # Service lines
    alley_x = int(w * 0.18)
    service_y_top = int(h * 0.28)
    service_y_bot = int(h * 0.72)

    # Singles sidelines
    cv2.line(image, (alley_x, margin_y), (alley_x, h - margin_y), color, thickness)
    cv2.line(image, (w - alley_x, margin_y), (w - alley_x, h - margin_y), color, thickness)

    # Service lines (horizontal)
    cv2.line(image, (alley_x, service_y_top), (w - alley_x, service_y_top), color, thickness)
    cv2.line(image, (alley_x, service_y_bot), (w - alley_x, service_y_bot), color, thickness)

    # Center service line
    center_x = w // 2
    cv2.line(image, (center_x, service_y_top), (center_x, net_y), color, thickness)
    cv2.line(image, (center_x, net_y), (center_x, service_y_bot), color, thickness)

    return image
