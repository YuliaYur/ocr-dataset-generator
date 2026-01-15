"""Perspective transform utility."""

import os

import cv2
import numpy as np


def perspective_transform(
    input_dir: str,
    filename: str,
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    x3: int,
    y3: int,
    x4: int,
    y4: int,
    output_dir: str = None,
) -> None:
    """Apply a perspective transform to a single image."""
    if not os.path.isdir(input_dir):
        raise ValueError('Invalid images directory path specified.')

    image_path = os.path.join(input_dir, filename)
    if not os.path.isfile(image_path):
        raise ValueError('Input image not found.')

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Unable to read input image.')

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result_image = cv2.warpPerspective(image, matrix, (width, height))

    if output_dir is None:
        output_dir = os.path.join(input_dir, 'perspective_transformed_images')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f'{os.path.splitext(filename)[0]}_perspective.jpg'
    )
    cv2.imwrite(output_path, result_image)
