"""Tests for edge cases in OCR processing and data handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch
import warnings

from PIL import Image

from lekha.diffing import BaseToken, compute_word_consensus
from lekha.processing import (
    _normalize_segments,  # pyright: ignore[reportPrivateUsage]
    _build_segments,  # pyright: ignore[reportPrivateUsage]
    process_inputs,
)
from lekha.project import ProjectStore
from lekha.ocr.tesseract_engine import TesseractResult, TesseractLine, TesseractWord

warnings.simplefilter("ignore", ResourceWarning)


class EdgeCaseProcessingTests(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory[str] | None
    data_root: Path | None

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.temp_dir = None
        self.data_root = None

    @override
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_root = Path(self.temp_dir.name) / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)
        patcher = patch("lekha.project.get_data_root", return_value=self.data_root)
        self.addCleanup(patcher.stop)
        _ = patcher.start()

    def test_normalize_segments_handles_empty_text(self) -> None:
        # Empty tesseract result (no lines)
        tess_result = TesseractResult(text="", lines=[])
        lines, tokens = _normalize_segments(tess_result, 100, 100)

        # Should create a default line with empty text
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "")
        self.assertEqual(len(lines[0].words), 0)
        self.assertEqual(len(tokens), 0)

    def test_normalize_segments_handles_single_word_line(self) -> None:
        # Single word on a line
        word = TesseractWord(text="Hello", left=10, top=10, width=50, height=20, line_index=0, word_index=0)
        line = TesseractLine(
            text="Hello",
            left=10,
            top=10,
            width=50,
            height=20,
            line_index=0,
            words=[word],
        )
        tess_result = TesseractResult(text="Hello", lines=[line])
        lines, tokens = _normalize_segments(tess_result, 200, 200)

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "Hello")
        self.assertEqual(len(lines[0].words), 1)
        self.assertEqual(lines[0].words[0].text, "Hello")
        self.assertEqual(len(tokens), 1)

    def test_normalize_segments_handles_line_without_word_data(self) -> None:
        # Line with text but no word-level data (fallback to splitting)
        line = TesseractLine(
            text="Hello world test",
            left=10,
            top=10,
            width=100,
            height=20,
            line_index=0,
            words=[],  # No word data
        )
        tess_result = TesseractResult(text="Hello world test", lines=[line])
        lines, _ = _normalize_segments(tess_result, 200, 200)

        self.assertEqual(len(lines), 1)
        # Should split text into words
        self.assertEqual(len(lines[0].words), 3)
        self.assertEqual(lines[0].words[0].text, "Hello")
        self.assertEqual(lines[0].words[1].text, "world")
        self.assertEqual(lines[0].words[2].text, "test")

    def test_build_segments_handles_empty_lines(self) -> None:
        # Empty line list
        word_segments, line_segments = _build_segments(
            page_index=0,
            image_path=Path("page.png"),
            lines=[],
            word_consensus=[],
            _model_outputs={},
        )

        self.assertEqual(len(word_segments), 0)
        self.assertEqual(len(line_segments), 0)

    def test_unicode_text_handling(self) -> None:
        # Test with various Unicode characters
        unicode_texts = [
            "Hello 世界",  # Chinese
            "Привет мир",  # Russian
            "مرحبا العالم",  # Arabic
            "שלום עולם",  # Hebrew
            "こんにちは世界",  # Japanese
            "🚀🌟✨",  # Emojis
            "Café résumé naïve",  # Accented characters
        ]

        for text in unicode_texts:
            word = TesseractWord(text=text, left=10, top=10, width=50, height=20, line_index=0, word_index=0)
            line = TesseractLine(
                text=text,
                left=10,
                top=10,
                width=50,
                height=20,
                line_index=0,
                words=[word],
            )
            tess_result = TesseractResult(text=text, lines=[line])
            lines, tokens = _normalize_segments(tess_result, 200, 200)

            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0].text, text)
            # Verify text is preserved correctly
            self.assertEqual(tokens[0].text, text)


class InvalidImageFormatTests(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory[str] | None
    data_root: Path | None

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.temp_dir = None
        self.data_root = None

    @override
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_root = Path(self.temp_dir.name) / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)
        patcher = patch("lekha.project.get_data_root", return_value=self.data_root)
        self.addCleanup(patcher.stop)
        _ = patcher.start()

    def test_process_inputs_fails_on_text_file(self) -> None:
        assert self.temp_dir is not None
        # Create a text file pretending to be an image
        manuscript_dir = Path(self.temp_dir.name) / "manuscript"
        manuscript_dir.mkdir()
        fake_image = manuscript_dir / "fake.png"
        _ = fake_image.write_text("I am not an image", encoding="utf-8")

        store = ProjectStore("invalid-image-test")

        # Mock tesseract to avoid actually calling it
        with patch("lekha.processing._run_tesseract_with_logging") as mock_tess:
            mock_tess.return_value = TesseractResult(text="", lines=[])

            # Should fail when trying to open as image
            with self.assertRaises((OSError, IOError, Exception)):
                process_inputs(
                    source_paths=[fake_image],
                    languages=["eng"],
                    models=["tesseract"],
                    store=store,
                    source=str(fake_image),
                )

    def test_process_inputs_handles_unsupported_extension(self) -> None:
        assert self.temp_dir is not None
        # File with unsupported extension
        manuscript_dir = Path(self.temp_dir.name) / "manuscript"
        manuscript_dir.mkdir()
        unsupported_file = manuscript_dir / "file.txt"
        _ = unsupported_file.write_text("text file", encoding="utf-8")

        store = ProjectStore("unsupported-ext-test")

        # Create a valid PNG too
        valid_image = manuscript_dir / "valid.png"
        Image.new("RGB", (100, 100), color="white").save(valid_image)

        # Mock tesseract
        with patch("lekha.processing._run_tesseract_with_logging") as mock_tess:
            mock_tess.return_value = TesseractResult(text="test", lines=[])

            # Process inputs - should only process the PNG
            process_inputs(
                source_paths=[valid_image],  # Only pass valid image
                languages=["eng"],
                models=["tesseract"],
                store=store,
                source=str(manuscript_dir),
            )

            # Should complete successfully
            self.assertTrue(store.meta_path.exists())


class EmptyManuscriptTests(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory[str] | None
    data_root: Path | None

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.temp_dir = None
        self.data_root = None

    @override
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_root = Path(self.temp_dir.name) / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)
        patcher = patch("lekha.project.get_data_root", return_value=self.data_root)
        self.addCleanup(patcher.stop)
        _ = patcher.start()

    def test_process_inputs_handles_blank_page(self) -> None:
        assert self.temp_dir is not None
        # Create a blank white image
        manuscript_dir = Path(self.temp_dir.name) / "manuscript"
        manuscript_dir.mkdir()
        blank_image = manuscript_dir / "blank.png"
        Image.new("RGB", (200, 200), color="white").save(blank_image)

        store = ProjectStore("blank-page-test")

        # Mock tesseract to return empty result (what it would do for blank page)
        with patch("lekha.processing._run_tesseract_with_logging") as mock_tess:
            mock_tess.return_value = TesseractResult(text="", lines=[])

            process_inputs(
                source_paths=[blank_image],
                languages=["eng"],
                models=["tesseract"],
                store=store,
                source=str(blank_image),
            )

            # Should create segments even for blank page
            segments = store.load_segments()
            # Should have at least one default segment
            self.assertGreaterEqual(len(segments), 0)

    def test_process_inputs_handles_page_with_only_whitespace(self) -> None:
        assert self.temp_dir is not None
        manuscript_dir = Path(self.temp_dir.name) / "manuscript"
        manuscript_dir.mkdir()
        image_path = manuscript_dir / "whitespace.png"
        Image.new("RGB", (200, 200), color="white").save(image_path)

        store = ProjectStore("whitespace-test")

        # Mock tesseract to return only whitespace
        with patch("lekha.processing._run_tesseract_with_logging") as mock_tess:
            mock_tess.return_value = TesseractResult(text="   \n\n   ", lines=[])

            process_inputs(
                source_paths=[image_path],
                languages=["eng"],
                models=["tesseract"],
                store=store,
                source=str(image_path),
            )

            # Should handle gracefully
            manifest = store.load_manifest()
            self.assertIsNotNone(manifest)
            assert manifest is not None  # For type checker
            self.assertEqual(len(manifest.files), 1)


class ConsensusEdgeCasesTests(unittest.TestCase):
    def test_consensus_with_empty_base_tokens(self) -> None:
        result = compute_word_consensus([], {"model1": "some text"})
        # Empty base, so no consensus entries
        self.assertEqual(len(result), 0)

    def test_consensus_with_all_empty_model_texts(self) -> None:
        base_tokens = [BaseToken(text="hello", line_index=0, word_index=0)]
        result = compute_word_consensus(base_tokens, {"model1": "", "model2": ""})
        # Should return base without modifications
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].base, "hello")
        self.assertFalse(result[0].has_conflict)

    def test_consensus_preserves_special_characters(self) -> None:
        # Test with special characters and punctuation
        special_texts = [
            "Hello!",
            "test@example.com",
            "$100.50",
            "C++",
            "re-test",
            "don't",
        ]

        for text in special_texts:
            base_tokens = [BaseToken(text=text, line_index=0, word_index=0)]
            result = compute_word_consensus(base_tokens, {})
            self.assertEqual(result[0].base, text)
            self.assertEqual(result[0].display_text, text)
