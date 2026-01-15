"""Basic geometric transformations for images."""

import cv2 as cv
import numpy as np


def resize(image: np.ndarray, width: int, height: int, interpolation=cv.INTER_LINEAR) -> np.ndarray:
    """Resize an image to a target width/height."""
    resized = cv.resize(image, (width, height), interpolation)
    return resized


def translate(image: np.ndarray, twidth: int, theight: int) -> np.ndarray:
    """Translate (shift) an image by (twidth, theight)."""
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, twidth], [0, 1, theight]])
    translated = cv.warpAffine(image, M, (cols, rows))
    return translated


def rotate(image: np.ndarray, angle: float, center: (int, int), border_value: int = 255) -> np.ndarray:
    """Rotate an image around a center and pad with a constant color."""
    height, width = image.shape[:2]
    x, y = center

    # M = cv.getRotationMatrix2D((x, y), angle, 1)

    theta = angle / 180.0 * np.pi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    M = np.float32([[cos_t, sin_t, x - x * cos_t - y * sin_t], [-sin_t, cos_t, y + x * sin_t - y * cos_t]])

    new_width = int(height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t + width * np.abs(sin_t))

    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y

    border = border_value
    if image.ndim == 3:
        border = (border_value,) * image.shape[2]

    rotated = cv.warpAffine(
        image,
        M,
        (new_width, new_height),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=border,
    )
    return rotated


def scale(image: np.ndarray, kwidth: float, kheight: float, center: (int, int)) -> np.ndarray:
    """Scale an image about a center point."""
    rows, cols = image.shape[:2]
    x, y = center
    M = np.float32([[kwidth, 0, x * (1 - kwidth)], [0, kheight, y * (1 - kheight)]])
    scaled = cv.warpAffine(image, M, (cols, rows))
    return scaled


def affine_transform(image: np.ndarray, pts1: np.float32, pts2: np.float32) -> np.ndarray:
    """Apply an affine transform to an image."""
    rows, cols = image.shape[:2]
    M = cv.getAffineTransform(pts1, pts2)
    transformed = cv.warpAffine(image, M, (cols, rows))
    return transformed




