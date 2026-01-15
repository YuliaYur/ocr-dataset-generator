"""Utilities for generating clear text images with annotations."""

import json
import os
import random
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _resolve_font(font_path: Optional[str], font_size: int) -> ImageFont.ImageFont:
    """Return a PIL font instance, falling back to common system fonts."""
    if font_path and os.path.isfile(font_path):
        return ImageFont.truetype(font=font_path, size=font_size)

    candidates = [
        os.path.join('C:\\', 'Windows', 'Fonts', 'arial.ttf'),
        os.path.join('C:\\', 'Windows', 'Fonts', 'calibri.ttf'),
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                return ImageFont.truetype(font=candidate, size=font_size)
            except OSError:
                continue

    return ImageFont.load_default()


def _text_size(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    """Return width/height of the text rendered with the given font."""
    if hasattr(font, 'getbbox'):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    return font.getsize(text)


def generate_clear_text_images(
    text_file_path: str,
    output_dir: str,
    num_images: int = 100,
    image_size: Tuple[int, int] = (256, 256),
    font_path: Optional[str] = None,
    font_size: int = 16,
    line_spacing: float = 1.5,
    border_margin: int = 4
) -> None:
    """Generate clear text images with word-level bounding boxes and corners.

    The output directory will contain an `images/` folder and `annotations.json`.
    """
    if not os.path.isfile(text_file_path):
        raise ValueError('Invalid text file path specified.')
    if num_images <= 0:
        raise ValueError('num_images must be a positive integer.')
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError('image_size must contain positive values.')

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    font = _resolve_font(font_path, font_size)
    width, height = image_size

    with open(text_file_path, 'r', encoding='utf-8') as handle:
        text = handle.read()
    words = text.split()
    if not words:
        raise ValueError('Text file is empty or contains no words.')

    space_width, _ = _text_size(font, ' ')
    annotations = {}
    word_index = 0

    for i in range(num_images):
        img = Image.new(mode='RGB', size=(width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        cursor_x = border_margin
        cursor_y = border_margin
        word_annotations = []
        can_place_text = True
        while can_place_text:
            if word_index >= len(words):
                word_index = 0

            word = words[word_index]
            word_width, word_height = _text_size(font, word)

            if cursor_x + word_width + border_margin > width:
                cursor_x = border_margin
                cursor_y += int(font_size * line_spacing)

            if cursor_y + word_height + border_margin > height:
                can_place_text = False
                break

            draw.text((cursor_x, cursor_y), word, (0, 0, 0), font=font)
            x1, y1 = cursor_x, cursor_y
            x2, y2 = cursor_x + word_width, cursor_y + word_height
            word_annotations.append({
                'word': word,
                'bbox': [x1, y1, x2, y2],
                'corners': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            })
            cursor_x += word_width + space_width
            word_index += 1

        file_name = f'clear_image_{i:05d}.png'
        img.save(os.path.join(images_dir, file_name))
        annotations[file_name] = {
            'width': width,
            'height': height,
            'words': word_annotations,
        }

    with open(os.path.join(output_dir, 'annotations.json'), 'w', encoding='utf-8') as write_file:
        json.dump(annotations, write_file, indent=4)
