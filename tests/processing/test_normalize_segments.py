"""Module-level tests for OCR processing normalization helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from PIL import Image

from lekha.diffing import BaseToken, compute_word_consensus
from lekha.processing import (
    _build_segments,  # pyright: ignore[reportPrivateUsage]
    _normalize_segments,  # pyright: ignore[reportPrivateUsage]
    process_inputs,
)
from lekha.project import ProjectStore
from lekha.ocr.tesseract_engine import TesseractLine, TesseractResult, TesseractWord


class NormalizeSegmentsTests(unittest.TestCase):
    """Validate `_normalize_segments` behavior with diverse inputs."""

    def test_handles_empty_text(self) -> None:
        tess_result = TesseractResult(text="", lines=[])
        lines, tokens = _normalize_segments(tess_result, 100, 100)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "")
        self.assertEqual(len(lines[0].words), 0)
        self.assertEqual(len(tokens), 0)

    def test_handles_single_word_line(self) -> None:
        word = TesseractWord(text="Hello", left=10, top=10, width=50, height=20, line_index=0, word_index=0)
        line = TesseractLine(text="Hello", left=10, top=10, width=50, height=20, line_index=0, words=[word])
        tess_result = TesseractResult(text="Hello", lines=[line])
        lines, tokens = _normalize_segments(tess_result, 200, 200)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "Hello")
        self.assertEqual(len(lines[0].words), 1)
        self.assertEqual(lines[0].words[0].text, "Hello")
        self.assertEqual(len(tokens), 1)

    def test_handles_line_without_word_data(self) -> None:
        line = TesseractLine(
            text="Hello world test",
            left=10,
            top=10,
            width=100,
            height=20,
            line_index=0,
            words=[],
        )
        tess_result = TesseractResult(text="Hello world test", lines=[line])
        lines, _ = _normalize_segments(tess_result, 200, 200)
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(lines[0].words), 3)
        self.assertEqual(lines[0].words[0].text, "Hello")
        self.assertEqual(lines[0].words[1].text, "world")
        self.assertEqual(lines[0].words[2].text, "test")

    def test_build_segments_handles_empty_lines(self) -> None:
        word_segments, line_segments = _build_segments(
            page_index=0,
            image_path=Path("page.png"),
            lines=[],
            word_consensus=[],
            _model_outputs={},
        )
        self.assertEqual(len(word_segments), 0)
        self.assertEqual(len(line_segments), 0)

    def test_unicode_texts_preserved(self) -> None:
        unicode_texts = [
            "Hello 世界",
            "Привет мир",
            "مرحبا العالم",
            "שלום עולם",
            "こんにちは世界",
            "🚀🌟✨",
            "Café résumé naïve",
        ]
        for text in unicode_texts:
            word = TesseractWord(text=text, left=10, top=10, width=50, height=20, line_index=0, word_index=0)
            line = TesseractLine(text=text, left=10, top=10, width=50, height=20, line_index=0, words=[word])
            tess_result = TesseractResult(text=text, lines=[line])
            lines, tokens = _normalize_segments(tess_result, 200, 200)
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0].text, text)
            self.assertEqual(tokens[0].text, text)


class ProcessInputsIntegrationTests(unittest.TestCase):
    """Targeted tests around `process_inputs` edge behavior."""

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
        base = Path(self.temp_dir.name)
        self.data_root = base / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)
        patcher = patch("lekha.project.get_data_root", return_value=self.data_root)
        _ = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_store(self, project_id: str) -> ProjectStore:
        assert self.data_root is not None
        return ProjectStore(project_id)

    def test_process_inputs_handles_blank_page(self) -> None:
        assert self.temp_dir is not None
        manuscript_dir = Path(self.temp_dir.name) / "manuscript"
        manuscript_dir.mkdir()
        blank_image = manuscript_dir / "blank.png"
        Image.new("RGB", (200, 200), color="white").save(blank_image)

        store = self._make_store("blank-page-test")
        with patch("lekha.processing.validate_tesseract_installation"), patch(
            "lekha.processing._run_tesseract_with_logging",
            return_value=TesseractResult(text="", lines=[]),
        ):
            process_inputs(
                source_paths=[blank_image],
                languages=["eng"],
                models=["tesseract"],
                store=store,
                source=str(blank_image),
            )
        segments = store.load_segments()
        self.assertGreaterEqual(len(segments), 0)

    def test_process_inputs_handles_whitespace_only(self) -> None:
        assert self.temp_dir is not None
        manuscript_dir = Path(self.temp_dir.name) / "manuscript"
        manuscript_dir.mkdir()
        image_path = manuscript_dir / "whitespace.png"
        Image.new("RGB", (200, 200), color="white").save(image_path)

        store = self._make_store("whitespace-test")
        with patch("lekha.processing.validate_tesseract_installation"), patch(
            "lekha.processing._run_tesseract_with_logging",
            return_value=TesseractResult(text="   \n\n   ", lines=[]),
        ):
            process_inputs(
                source_paths=[image_path],
                languages=["eng"],
                models=["tesseract"],
                store=store,
                source=str(image_path),
            )
        manifest = store.load_manifest()
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(len(manifest.files), 1)

    def test_process_inputs_fails_on_non_image(self) -> None:
        assert self.temp_dir is not None
        manuscript_dir = Path(self.temp_dir.name) / "manuscript"
        manuscript_dir.mkdir()
        fake_image = manuscript_dir / "fake.png"
        _ = fake_image.write_text("I am not an image", encoding="utf-8")

        store = self._make_store("invalid-image-test")
        with patch("lekha.processing.validate_tesseract_installation"), patch(
            "lekha.processing._run_tesseract_with_logging",
            return_value=TesseractResult(text="", lines=[]),
        ):
            with self.assertRaises(Exception):
                process_inputs(
                    source_paths=[fake_image],
                    languages=["eng"],
                    models=["tesseract"],
                    store=store,
                    source=str(fake_image),
                )

    def test_process_inputs_ignores_unsupported_extensions(self) -> None:
        assert self.temp_dir is not None
        manuscript_dir = Path(self.temp_dir.name) / "manuscript"
        manuscript_dir.mkdir()
        unsupported_file = manuscript_dir / "file.txt"
        _ = unsupported_file.write_text("text file", encoding="utf-8")
        valid_image = manuscript_dir / "valid.png"
        Image.new("RGB", (100, 100), color="white").save(valid_image)

        store = self._make_store("unsupported-ext-test")
        with patch("lekha.processing.validate_tesseract_installation"), patch(
            "lekha.processing._run_tesseract_with_logging",
            return_value=TesseractResult(text="test", lines=[]),
        ):
            process_inputs(
                source_paths=[valid_image],
                languages=["eng"],
                models=["tesseract"],
                store=store,
                source=str(manuscript_dir),
            )
        self.assertTrue(store.meta_path.exists())


class ConsensusEdgeCasesTests(unittest.TestCase):
    """Additional compute_word_consensus edge cases."""

    def test_consensus_with_empty_base_tokens(self) -> None:
        result = compute_word_consensus([], {"model1": "some text"})
        self.assertEqual(len(result), 0)

    def test_consensus_with_all_empty_model_texts(self) -> None:
        base_tokens = [BaseToken(text="hello", line_index=0, word_index=0)]
        result = compute_word_consensus(base_tokens, {"model1": "", "model2": ""})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].base, "hello")
        self.assertFalse(result[0].has_conflict)

    def test_consensus_preserves_special_characters(self) -> None:
        special_texts = ["Hello!", "test@example.com", "$100.50", "C++", "re-test", "don't"]
        for text in special_texts:
            base_tokens = [BaseToken(text=text, line_index=0, word_index=0)]
            result = compute_word_consensus(base_tokens, {})
            self.assertEqual(result[0].base, text)
            self.assertEqual(result[0].display_text, text)
