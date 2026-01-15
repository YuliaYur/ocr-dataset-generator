import shutil
import unittest
from pathlib import Path

from src.generate_clear_text_images import generate_clear_text_images
from src.downscaled_image_generator import generate_downscaled_images
from src.degrade_dataset import generate_degraded_dataset
from definitions import ROOT_DIR


class SmokeTest(unittest.TestCase):
    def test_generate_clear_text_images(self) -> None:
        out_dir = Path('tmp') / 'text'
        generate_clear_text_images(
            text_file_path=ROOT_DIR / 'data/The_Picture_of_Dorian_Gray.txt',
            output_dir=str(out_dir),
            num_images=1
        )

        images_dir = out_dir / 'images'
        assert images_dir.is_dir()
        assert any(images_dir.iterdir())
        assert (out_dir / 'annotations.json').is_file()


    def test_generate_downscaled_images(self) -> None:
        images_dir =  Path('tmp') / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT_DIR / 'data/sample_images/text_image.png', images_dir / 'text_image.png')

        generate_downscaled_images(
            images_dir=str(images_dir),
            target_size=(64, 64),
        )

        downscaled_dir = Path('tmp') / 'downscaled_images'
        assert downscaled_dir.is_dir()
        assert any(downscaled_dir.iterdir())


    def test_generate_degraded_dataset(self) -> None:
        clear_dir = Path('tmp') / 'clear'
        generate_clear_text_images(
            text_file_path=ROOT_DIR / 'data/The_Picture_of_Dorian_Gray.txt',
            output_dir=str(clear_dir),
            num_images=1
        )

        images_dir = clear_dir / 'images'
        annotations_path = clear_dir / 'annotations.json'
        degraded_dir = Path('tmp') / 'degraded'

        generate_degraded_dataset(
            images_dir=str(images_dir),
            annotations_path=str(annotations_path),
            output_dir=str(degraded_dir),
            num_images=1,
            seed=1,
            max_rotate=0.0,
            run_tesseract=False,
        )

        degraded_images_dir = degraded_dir / 'images'
        assert degraded_images_dir.is_dir()
        assert any(degraded_images_dir.iterdir())
