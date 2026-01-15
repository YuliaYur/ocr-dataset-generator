"""Geometry helpers for point transformations."""

from typing import Tuple

import numpy as np

Point2D = Tuple[int, int]
Size2D = Tuple[int, int]


def scale_point2d(src_point: Point2D, original_size: Size2D, target_size: Size2D) -> Point2D:
    """Scale a point from one image size to another."""
    orig_width, orig_height = original_size
    target_width, target_height = target_size

    src_x, src_y = src_point
    scaled_x = src_x / orig_width * target_width
    scaled_y = src_y / orig_height * target_height

    return round(scaled_x), round(scaled_y)


def rotate_point2d(src_point: Point2D, angle: float, center: Point2D, radians: bool = False) -> Point2D:
    """Rotate a point around a center by the given angle."""
    angle_rad = angle if radians else angle * np.pi / 180

    (x1, y1), (x0, y0) = src_point, center
    rotated_x = (x1 - x0) * np.cos(angle_rad) + (y1 - y0) * np.sin(angle_rad) + x0
    rotated_y = -(x1 - x0) * np.sin(angle_rad) + (y1 - y0) * np.cos(angle_rad) + y0

    return round(rotated_x), round(rotated_y)


def rotate_point2d_no_crop(src_point: Point2D, angle: float, center: Point2D, img_size: Size2D) -> Point2D:
    """Rotate a point using the expanded canvas size (no cropping)."""
    width, height = img_size
    center_x, center_y = center
    point_x, point_y = src_point

    theta = angle / 180.0 * np.pi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    rotated_x = (point_x - center_x) * cos_t + (point_y - center_y) * sin_t + center_x
    rotated_y = -(point_x - center_x) * sin_t + (point_y - center_y) * cos_t + center_y

    new_width = int(height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t + width * np.abs(sin_t))

    rotated_x += (new_width / 2) - center_x
    rotated_y += (new_height / 2) - center_y

    return round(rotated_x), round(rotated_y)
