"""Blur utilities for image datasets."""

import glob
import os

import PIL
import numpy as np
from PIL import Image, ImageFilter


_FILTERS = {
    'gaussian': lambda image, radius: image.filter(ImageFilter.GaussianBlur(radius)),
    'box': lambda image, radius: image.filter(ImageFilter.BoxBlur(radius)),
    'min': lambda image, radius: image.filter(ImageFilter.MinFilter(radius)),
    'max': lambda image, radius: image.filter(ImageFilter.MaxFilter(radius)),
    'median': lambda image, radius: image.filter(ImageFilter.MedianFilter(radius)),
}


def blur_images(input_dir: str, filename: str, filter_name: str, radius: int, output_dir: str = None) -> None:
    """Blur images in a directory with a selected filter."""
    if not os.path.isdir(input_dir):
        raise ValueError('Invalid images directory path specified.')

    filter_key = (filter_name or '').lower()
    if filter_key not in _FILTERS:
        raise ValueError(f'Invalid filter specified: {filter_name}')

    if radius <= 0:
        raise ValueError('Radius must be a positive integer.')

    if output_dir is None:
        output_dir = os.path.join(input_dir, 'blurred_images')
    os.makedirs(output_dir, exist_ok=True)

    if filename:
        candidates = [os.path.join(input_dir, filename)]
    else:
        candidates = glob.glob(os.path.join(input_dir, '*.*'))

    if not candidates:
        raise ValueError('No images found to process.')

    processed = 0
    for image_path in candidates:
        if not os.path.isfile(image_path):
            continue
        image = Image.open(image_path)
        result_image = _FILTERS[filter_key](image, radius)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_{filter_key}_r{radius}.png')
        result_image.save(output_path)
        processed += 1

    if processed == 0:
        raise ValueError('No valid images found to process.')


def gaussian_blur(image: np.array, radius=1) -> np.array:
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(blured)


def box_blur(image: np.array, radius=1) -> np.array:
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.BoxBlur(radius))
    return np.asarray(blured)


def min_filter(image: np.array, radius=3) -> np.array:
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.MinFilter(radius))
    return np.asarray(blured)


def max_filter(image: np.array, radius=3) -> np.array:
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.MaxFilter(radius))
    return np.asarray(blured)


def median_filter(image: np.array, radius=3) -> np.array:
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.MedianFilter(radius))
    return np.asarray(blured)
