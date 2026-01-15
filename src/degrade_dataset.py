"""Dataset degradation pipeline for OCR training data."""

import json
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import cv2 as cv
import numpy as np
import pytesseract as tesseract
from PIL import Image

from .image_ops import (
    BoxBlurOperation,
    GaussianBlurOperation,
    GaussianNoiseOperation,
    MaxFilterOperation,
    MinFilterOperation,
    ResizeOperation,
    RotateOperation,
    SaltPepperOperation,
    SpeckleOperation,
)
from .metrics import calculate_relative_edit_distance
from .utils import scale_point2d, rotate_point2d_no_crop


def _load_annotations(path: str) -> Dict:
    """Load annotation data from JSON."""
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _get_word_corners(word_data: Dict) -> List[Tuple[int, int]]:
    """Return a list of corner points from supported annotation formats."""
    corners = word_data.get('corners')
    if corners:
        return [tuple(point) for point in corners]

    quad = word_data.get('quad')
    if quad:
        return [tuple(point) for point in quad]

    bbox = word_data.get('bbox')
    if not bbox:
        if all(key in word_data for key in ('x1', 'y1', 'x2', 'y2')):
            bbox = [word_data['x1'], word_data['y1'], word_data['x2'], word_data['y2']]
        else:
            return []

    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _extract_image_text(annotations: Dict, image_name: str) -> str:
    """Build the full text string from word annotations."""
    result = ''
    image_data = annotations[image_name]
    words = image_data.get('words', image_data)
    for word_data in words:
        word = str(word_data.get('word', ''))
        if '\n' not in word:
            result += word + ' '
        else:
            result += word
    result += '\f'
    return result


def _project_word_boxes(
    annotations: Dict,
    image_name: str,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    angle: float,
) -> List[Dict]:
    """Project word boxes into the degraded image space."""
    word_boxes = []
    target_width, target_height = target_size
    image_data = annotations[image_name]
    words_data = image_data.get('words', image_data)
    for word_data in words_data:
        word = word_data.get('word', '')
        source_corners = _get_word_corners(word_data)
        if not source_corners:
            continue

        projected_corners = []
        for point in source_corners:
            scaled_point = scale_point2d(src_point=point, original_size=source_size, target_size=target_size)
            new_point = rotate_point2d_no_crop(
                src_point=scaled_point,
                angle=angle,
                center=(target_width // 2, target_height // 2),
                img_size=target_size,
            )
            projected_corners.append(new_point)

        xs = [p[0] for p in projected_corners]
        ys = [p[1] for p in projected_corners]
        word_boxes.append({
            'word': word,
            'corners': projected_corners,
            'bbox': [min(xs), min(ys), max(xs), max(ys)],
        })
    return word_boxes


def _validate_degradation_config(
    resize_min: float,
    resize_max: float,
    gaussian_mean_min: float,
    gaussian_mean_max: float,
    gaussian_std_min: float,
    gaussian_std_max: float,
    speckle_std: float,
    speckle_std_alt: float,
    salt_vs_pepper_min: float,
    salt_vs_pepper_max: float,
    salt_pepper_amount_min: float,
    salt_pepper_amount_max: float,
) -> None:
    if resize_min <= 0 or resize_max <= 0 or resize_min > resize_max:
        raise ValueError('Invalid resize range specified.')
    if gaussian_mean_min > gaussian_mean_max:
        raise ValueError('Invalid gaussian mean range specified.')
    if gaussian_std_min > gaussian_std_max or gaussian_std_min < 0 or gaussian_std_max < 0:
        raise ValueError('Invalid gaussian stddev range specified.')
    if speckle_std < 0 or speckle_std_alt < 0:
        raise ValueError('Invalid speckle stddev specified.')
    if salt_vs_pepper_min > salt_vs_pepper_max:
        raise ValueError('Invalid salt-vs-pepper range specified.')
    if salt_pepper_amount_min > salt_pepper_amount_max or salt_pepper_amount_min < 0 or salt_pepper_amount_max < 0:
        raise ValueError('Invalid salt-pepper amount range specified.')


def _pick_radius(min_value: int, max_value: int, step: int) -> int:
    """Pick an integer radius from a range with a fixed step."""
    if step <= 0:
        raise ValueError('Radius step must be a positive integer.')
    if min_value > max_value:
        raise ValueError('Invalid radius range specified.')
    return random.randrange(min_value, max_value + 1, step)


def _apply_operations(image: np.ndarray, operations: Iterable) -> np.ndarray:
    """Apply a sequence of operations to an image."""
    result = image
    for operation in operations:
        result = operation(result)
    return result


def generate_degraded_dataset(
    images_dir: str,
    annotations_path: str,
    output_dir: str,
    num_images: Optional[int] = None,
    seed: Optional[int] = None,
    max_rotate: float = 5.0,
    resize_min: float = 0.66,
    resize_max: float = 1.5,
    use_gaussian_noise: bool = True,
    use_speckle: bool = True,
    use_salt_pepper: bool = True,
    use_gaussian_blur: bool = True,
    use_box_blur: bool = True,
    use_max_filter: bool = True,
    use_min_filter: bool = True,
    use_resize: bool = True,
    use_rotate: bool = True,
    gaussian_mean_min: float = 0.5,
    gaussian_mean_max: float = 0.9,
    gaussian_std_min: float = 0.05,
    gaussian_std_max: float = 0.09,
    speckle_mean: float = 0.0,
    speckle_std: float = 0.001,
    speckle_mean_alt: float = 1.0,
    speckle_std_alt: float = 0.0,
    salt_vs_pepper_min: float = 0.0,
    salt_vs_pepper_max: float = 1.0,
    salt_pepper_amount_min: float = 0.0,
    salt_pepper_amount_max: float = 0.02,
    gaussian_blur_radius_min: int = 0,
    gaussian_blur_radius_max: int = 2,
    box_blur_radius_min: int = 1,
    box_blur_radius_max: int = 2,
    max_filter_radius_min: int = 1,
    max_filter_radius_max: int = 3,
    min_filter_radius_min: int = 1,
    min_filter_radius_max: int = 3,
    tesseract_cmd: Optional[str] = None,
    run_tesseract: bool = True,
) -> None:
    """Generate degraded images and matching annotations from clear text images."""
    if not os.path.isdir(images_dir):
        raise ValueError('Invalid images directory path specified.')
    if not os.path.isfile(annotations_path):
        raise ValueError('Invalid annotations path specified.')
    _validate_degradation_config(
        resize_min=resize_min,
        resize_max=resize_max,
        gaussian_mean_min=gaussian_mean_min,
        gaussian_mean_max=gaussian_mean_max,
        gaussian_std_min=gaussian_std_min,
        gaussian_std_max=gaussian_std_max,
        speckle_std=speckle_std,
        speckle_std_alt=speckle_std_alt,
        salt_vs_pepper_min=salt_vs_pepper_min,
        salt_vs_pepper_max=salt_vs_pepper_max,
        salt_pepper_amount_min=salt_pepper_amount_min,
        salt_pepper_amount_max=salt_pepper_amount_max,
    )

    os.makedirs(output_dir, exist_ok=True)
    images_out_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_out_dir, exist_ok=True)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if tesseract_cmd:
        tesseract.pytesseract.tesseract_cmd = tesseract_cmd

    annotation_data = _load_annotations(annotations_path)
    image_filenames = sorted(annotation_data.keys())
    if num_images is not None:
        image_filenames = image_filenames[:num_images]

    interpolations = [cv.INTER_AREA, cv.INTER_LINEAR, cv.INTER_CUBIC]
    output_index = {}

    for idx, image_name in enumerate(image_filenames):
        image_path = os.path.join(images_dir, image_name)
        if not os.path.isfile(image_path):
            continue

        source_image = cv.imread(image_path)
        if source_image is None:
            continue

        source_height, source_width = source_image.shape[:2]
        source_size = (source_width, source_height)
        if use_resize:
            target_width = max(1, int(source_width * random.uniform(resize_min, resize_max)))
            target_height = max(1, int(source_height * random.uniform(resize_min, resize_max)))
        else:
            target_width = source_width
            target_height = source_height
        target_size = (target_width, target_height)

        degradation_ops = []
        if use_gaussian_noise:
            degradation_ops.append(
                GaussianNoiseOperation(
                    mean=random.uniform(gaussian_mean_min, gaussian_mean_max),
                    stddev=random.uniform(gaussian_std_min, gaussian_std_max),
                )
            )
        if use_speckle:
            degradation_ops.append(SpeckleOperation(mean=speckle_mean, stddev=speckle_std))
            degradation_ops.append(SpeckleOperation(mean=speckle_mean_alt, stddev=speckle_std_alt))
        if use_salt_pepper:
            degradation_ops.append(
                SaltPepperOperation(
                    salt_vs_pepper=random.uniform(salt_vs_pepper_min, salt_vs_pepper_max),
                    amount=random.uniform(salt_pepper_amount_min, salt_pepper_amount_max),
                )
            )
        if use_gaussian_blur:
            degradation_ops.append(GaussianBlurOperation(
                radius=_pick_radius(gaussian_blur_radius_min, gaussian_blur_radius_max, 2)
            ))
        if use_box_blur:
            degradation_ops.append(BoxBlurOperation(
                radius=_pick_radius(box_blur_radius_min, box_blur_radius_max, 1)
            ))
        if use_max_filter:
            degradation_ops.append(MaxFilterOperation(
                radius=_pick_radius(max_filter_radius_min, max_filter_radius_max, 2)
            ))
        if use_min_filter:
            degradation_ops.append(MinFilterOperation(
                radius=_pick_radius(min_filter_radius_min, min_filter_radius_max, 2)
            ))
        if use_resize:
            degradation_ops.append(
                ResizeOperation(
                    width=target_width,
                    height=target_height,
                    interpolation=interpolations[random.randint(0, 2)],
                )
            )

        degraded_image = _apply_operations(source_image, degradation_ops) if degradation_ops else source_image
        resized_to_source = ResizeOperation(
            width=source_width,
            height=source_height,
            interpolation=cv.INTER_CUBIC,
        )(degraded_image)
        psnr_value = float(cv.PSNR(source_image, resized_to_source))

        if use_rotate:
            angle = random.uniform(-max_rotate, max_rotate)
            degraded = RotateOperation(angle=angle, center=(target_width // 2, target_height // 2))(degraded_image)
        else:
            angle = 0.0
            degraded = degraded_image
        degraded_gray = cv.cvtColor(degraded, cv.COLOR_BGR2GRAY)

        output_name = f'degraded_{idx:05d}.png'
        Image.fromarray(np.uint8(degraded_gray)).save(os.path.join(images_out_dir, output_name))

        tesseract_text = None
        tesseract_relative_error = None
        if run_tesseract:
            real_text = _extract_image_text(annotation_data, image_name)
            tesseract_text = tesseract.image_to_string(degraded_gray)
            tesseract_relative_error = int(calculate_relative_edit_distance(real_text, tesseract_text))

        word_boxes = _project_word_boxes(
            annotations=annotation_data,
            image_name=image_name,
            source_size=source_size,
            target_size=target_size,
            angle=angle,
        )

        output_index[output_name] = {
            'source_image': image_name,
            'width': degraded_gray.shape[1],
            'height': degraded_gray.shape[0],
            'psnr': psnr_value,
            'tesseract_output': (tesseract_text.split('\n') if tesseract_text else None),
            'tesseract_relative_error': tesseract_relative_error,
            'words': word_boxes,
        }

    with open(os.path.join(output_dir, 'annotations.json'), 'w', encoding='utf-8') as write_file:
        json.dump(output_index, write_file, indent=4)
