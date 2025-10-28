from __future__ import annotations

import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from lekha.ocr.tesseract_engine import run_tesseract, validate_tesseract_installation

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


class TesseractValidationTests(unittest.TestCase):
    def test_validate_tesseract_installation_succeeds_when_installed(self) -> None:
        # This test will only pass if tesseract is actually installed
        # If pytesseract is available, the validation should succeed
        import sys

        # Temporarily make pytesseract appear missing
        pytesseract_module = sys.modules.get("pytesseract")
        if pytesseract_module is None:
            # pytesseract not installed, validation should fail
            with self.assertRaises(RuntimeError) as ctx:
                validate_tesseract_installation()
            self.assertIn("pytesseract is not installed", str(ctx.exception))
        else:
            # pytesseract is installed, should succeed or fail based on tesseract availability
            try:
                validate_tesseract_installation()
            except RuntimeError as exc:
                # If it fails, it should have helpful error message
                error_msg = str(exc)
                self.assertTrue(
                    "not installed" in error_msg or "not accessible" in error_msg,
                    f"Expected helpful error message, got: {error_msg}",
                )

    def test_validate_tesseract_installation_fails_when_pytesseract_missing(self) -> None:
        with patch("lekha.ocr.tesseract_engine.pytesseract", None):
            with self.assertRaises(RuntimeError) as ctx:
                validate_tesseract_installation()
            error_msg = str(ctx.exception)
            self.assertIn("pytesseract is not installed", error_msg)
            self.assertIn("pip install pytesseract", error_msg)

    def test_validate_tesseract_installation_fails_when_tesseract_not_accessible(self) -> None:
        # Create a non-None mock for pytesseract to ensure we reach the subprocess check
        mock_pytesseract = object()
        # Mock subprocess.run to simulate tesseract not being found
        with patch("lekha.ocr.tesseract_engine.pytesseract", new=mock_pytesseract), patch(
            "lekha.ocr.tesseract_engine.subprocess.run"
        ) as mock_run:
            mock_run.side_effect = FileNotFoundError("tesseract not found in PATH")
            with self.assertRaises(RuntimeError) as ctx:
                validate_tesseract_installation()
            error_msg = str(ctx.exception)
            self.assertIn("not installed or not accessible", error_msg)
            self.assertIn("brew install tesseract", error_msg)
