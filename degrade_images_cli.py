"""CLI entrypoint for generating degraded OCR images."""

import argparse
import os
from typing import List, Optional

from src.degrade_dataset import generate_degraded_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate degraded OCR images with annotations.')
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--annotations')
    parser.add_argument('--out-dir', default=os.path.join('output', 'degraded'))
    parser.add_argument('--num-images', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-rotate', type=float, default=5.0)
    parser.add_argument('--resize-min', type=float, default=0.66)
    parser.add_argument('--resize-max', type=float, default=1.5)
    parser.add_argument('--gaussian-mean-min', type=float, default=0.5)
    parser.add_argument('--gaussian-mean-max', type=float, default=0.9)
    parser.add_argument('--gaussian-std-min', type=float, default=0.05)
    parser.add_argument('--gaussian-std-max', type=float, default=0.09)
    parser.add_argument('--speckle-mean', type=float, default=0.0)
    parser.add_argument('--speckle-std', type=float, default=0.001)
    parser.add_argument('--speckle-mean-alt', type=float, default=1.0)
    parser.add_argument('--speckle-std-alt', type=float, default=0.0)
    parser.add_argument('--salt-vs-pepper-min', type=float, default=0.0)
    parser.add_argument('--salt-vs-pepper-max', type=float, default=1.0)
    parser.add_argument('--salt-pepper-amount-min', type=float, default=0.0)
    parser.add_argument('--salt-pepper-amount-max', type=float, default=0.02)
    parser.add_argument('--gaussian-blur-radius-min', type=int, default=0)
    parser.add_argument('--gaussian-blur-radius-max', type=int, default=2)
    parser.add_argument('--box-blur-radius-min', type=int, default=1)
    parser.add_argument('--box-blur-radius-max', type=int, default=2)
    parser.add_argument('--max-filter-radius-min', type=int, default=1)
    parser.add_argument('--max-filter-radius-max', type=int, default=3)
    parser.add_argument('--min-filter-radius-min', type=int, default=1)
    parser.add_argument('--min-filter-radius-max', type=int, default=3)
    parser.add_argument('--no-gaussian-noise', dest='use_gaussian_noise', action='store_false', help='Disable gaussian noise.')
    parser.add_argument('--no-speckle', dest='use_speckle', action='store_false', help='Disable speckle noise.')
    parser.add_argument('--no-salt-pepper', dest='use_salt_pepper', action='store_false', help='Disable salt-and-pepper noise.')
    parser.add_argument('--no-gaussian-blur', dest='use_gaussian_blur', action='store_false', help='Disable gaussian blur.')
    parser.add_argument('--no-box-blur', dest='use_box_blur', action='store_false', help='Disable box blur.')
    parser.add_argument('--no-max-filter', dest='use_max_filter', action='store_false', help='Disable max filter.')
    parser.add_argument('--no-min-filter', dest='use_min_filter', action='store_false', help='Disable min filter.')
    parser.add_argument('--no-resize', dest='use_resize', action='store_false', help='Disable resize operation.')
    parser.add_argument('--no-rotate', dest='use_rotate', action='store_false', help='Disable rotation.')
    parser.set_defaults(
        use_gaussian_noise=True,
        use_speckle=True,
        use_salt_pepper=True,
        use_gaussian_blur=True,
        use_box_blur=True,
        use_max_filter=True,
        use_min_filter=True,
        use_resize=True,
        use_rotate=True,
    )

    parser.add_argument('--tesseract-cmd')
    parser.add_argument('--skip-tesseract', action='store_true')
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    annotations = args.annotations
    if not annotations:
        annotations = os.path.join(os.path.dirname(args.images_dir), 'annotations.json')

    generate_degraded_dataset(
        images_dir=args.images_dir,
        annotations_path=annotations,
        output_dir=args.out_dir,
        num_images=args.num_images,
        seed=args.seed,
        max_rotate=args.max_rotate,
        resize_min=args.resize_min,
        resize_max=args.resize_max,
        use_gaussian_noise=args.use_gaussian_noise,
        use_speckle=args.use_speckle,
        use_salt_pepper=args.use_salt_pepper,
        use_gaussian_blur=args.use_gaussian_blur,
        use_box_blur=args.use_box_blur,
        use_max_filter=args.use_max_filter,
        use_min_filter=args.use_min_filter,
        use_resize=args.use_resize,
        use_rotate=args.use_rotate,
        gaussian_mean_min=args.gaussian_mean_min,
        gaussian_mean_max=args.gaussian_mean_max,
        gaussian_std_min=args.gaussian_std_min,
        gaussian_std_max=args.gaussian_std_max,
        speckle_mean=args.speckle_mean,
        speckle_std=args.speckle_std,
        speckle_mean_alt=args.speckle_mean_alt,
        speckle_std_alt=args.speckle_std_alt,
        salt_vs_pepper_min=args.salt_vs_pepper_min,
        salt_vs_pepper_max=args.salt_vs_pepper_max,
        salt_pepper_amount_min=args.salt_pepper_amount_min,
        salt_pepper_amount_max=args.salt_pepper_amount_max,
        gaussian_blur_radius_min=args.gaussian_blur_radius_min,
        gaussian_blur_radius_max=args.gaussian_blur_radius_max,
        box_blur_radius_min=args.box_blur_radius_min,
        box_blur_radius_max=args.box_blur_radius_max,
        max_filter_radius_min=args.max_filter_radius_min,
        max_filter_radius_max=args.max_filter_radius_max,
        min_filter_radius_min=args.min_filter_radius_min,
        min_filter_radius_max=args.min_filter_radius_max,
        tesseract_cmd=args.tesseract_cmd,
        run_tesseract=not args.skip_tesseract,
    )


if __name__ == '__main__':
    main()
