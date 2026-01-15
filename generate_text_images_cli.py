"""CLI entrypoint for generating clear text images."""

import argparse
import os
from typing import List, Optional

from src.generate_clear_text_images import generate_clear_text_images


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate clear text images with annotations.')
    parser.add_argument('--text-file', required=True)
    parser.add_argument('--out-dir', default=os.path.join('output', 'text'))
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--image-width', type=int, default=256)
    parser.add_argument('--image-height', type=int, default=256)
    parser.add_argument('--font-size', type=int, default=16)
    parser.add_argument('--line-spacing', type=float, default=1.5)
    parser.add_argument('--border-margin', type=int, default=4)
    parser.add_argument('--font-path')
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    generate_clear_text_images(
        text_file_path=args.text_file,
        output_dir=args.out_dir,
        num_images=args.num_images,
        image_size=(args.image_width, args.image_height),
        font_path=args.font_path,
        font_size=args.font_size,
        line_spacing=args.line_spacing
    )


if __name__ == '__main__':
    main()
