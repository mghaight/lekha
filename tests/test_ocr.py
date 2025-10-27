from __future__ import annotations

import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from lekha.ocr.tesseract_engine import run_tesseract

warnings.simplefilter("ignore", ResourceWarning)


class OCRTests(unittest.TestCase):
    def test_run_tesseract_parses_structured_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            Image.new("RGB", (100, 40), color="white").save(image_path)

            dummy_output = types.SimpleNamespace(DICT="DICT")

            class DummyPytesseract:
                @staticmethod
                def image_to_data(image: Image.Image, lang: str, output_type: object) -> dict[str, list[str]]:
                    assert image and lang and output_type
                    return {
                        "text": ["Hello", "World"],
                        "line_num": ["1", "1"],
                        "block_num": ["1", "1"],
                        "par_num": ["1", "1"],
                        "left": ["0", "50"],
                        "top": ["0", "0"],
                        "width": ["40", "40"],
                        "height": ["10", "10"],
                    }

            with patch("lekha.ocr.tesseract_engine.pytesseract", new=DummyPytesseract()), patch(
                "lekha.ocr.tesseract_engine.Output", new=dummy_output
            ):
                result = run_tesseract(image_path, ["eng"])

        self.assertEqual(result.text, "Hello World")
        self.assertEqual(len(result.lines), 1)
        self.assertEqual(len(result.lines[0].words), 2)

    def test_run_tesseract_raises_runtime_error_on_tesseract_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            Image.new("RGB", (10, 10), color="white").save(image_path)

            class FakeError(Exception):
                pass

            class FailingPytesseract:
                @staticmethod
                def image_to_data(*args: object, **kwargs: object) -> dict[str, list[str]]:
                    assert args or kwargs
                    raise FakeError("failed loading language data")

            with patch("lekha.ocr.tesseract_engine.pytesseract", new=FailingPytesseract()), patch(
                "lekha.ocr.tesseract_engine.Output", new=types.SimpleNamespace(DICT="DICT")
            ), patch("lekha.ocr.tesseract_engine.TesseractError", new=FakeError):
                with self.assertRaises(RuntimeError) as ctx:
                    _ = run_tesseract(image_path, ["san"])

        self.assertIn("missing the traineddata", str(ctx.exception))
