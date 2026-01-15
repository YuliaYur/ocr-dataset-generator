"""Utilities for downscaling image datasets."""

import glob
import os
from typing import Optional, Tuple

import cv2
from PIL import Image


def _get_interpolation_key(interpolation: str) -> int:
    if interpolation == 'nearest':
        return cv2.INTER_NEAREST

    if interpolation == 'linear':
        return cv2.INTER_LINEAR

    if interpolation == 'area':
        return cv2.INTER_AREA

    return cv2.INTER_CUBIC


def generate_downscaled_images(
    images_dir: str,
    target_size: Tuple[int, int],
    interpolation: str = 'cubic',
    output_dir: Optional[str] = None,
) -> None:
    """Resize images in a directory to a target size."""
    if not os.path.isdir(images_dir):
        raise ValueError('Invalid images directory path specified.')
    
    if not all(target_size):
        raise ValueError('Invalid image target size specified.')

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(images_dir), 'downscaled_images')
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    inter_key = _get_interpolation_key(interpolation)
    for fp in glob.glob(os.path.join(images_dir, '*.*')):
        try:
            raw = cv2.imread(fp)
            if raw is None:
                raise ValueError('Image could not be read.')
            image = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            target_fp = os.path.join(output_dir, os.path.split(fp)[-1])
            resized = cv2.resize(image, target_size, interpolation=inter_key)
            Image.fromarray(resized).save(target_fp)
        except Exception as ex:
            print(f'Couldn\'t resize {fp}. {ex}')
