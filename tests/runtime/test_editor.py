"""Unit tests for `SegmentEditor` behaviors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from lekha.project import ProjectStore, Segment
from lekha.runtime.editor import SegmentEditor


def _sample_segments() -> list[Segment]:
    line_segment = Segment(
        segment_id="p000_l0000",
        view="line",
        page_index=0,
        line_index=0,
        word_index=None,
        page_image="page.png",
        bbox={"x": 10, "y": 10, "w": 80, "h": 20},
        base_text="base line",
        consensus_text="hello world",
        has_conflict=False,
        alternatives={},
        word_ids=["p000_l0000_w0000", "p000_l0000_w0001"],
    )
    word_one = Segment(
        segment_id="p000_l0000_w0000",
        view="word",
        page_index=0,
        line_index=0,
        word_index=0,
        page_image="page.png",
        bbox={"x": 10, "y": 10, "w": 30, "h": 18},
        base_text="hello",
        consensus_text="hello",
        has_conflict=False,
    )
    word_two = Segment(
        segment_id="p000_l0000_w0001",
        view="word",
        page_index=0,
        line_index=0,
        word_index=1,
        page_image="page.png",
        bbox={"x": 45, "y": 10, "w": 30, "h": 18},
        base_text="world",
        consensus_text="world",
        has_conflict=False,
    )
    return [line_segment, word_one, word_two]


class SegmentEditorTests(unittest.TestCase):
    """Direct unit tests for `SegmentEditor` helper logic."""

    temp_dir: tempfile.TemporaryDirectory[str] | None
    data_root: Path | None
    store: ProjectStore | None
    editor: SegmentEditor | None

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.temp_dir = None
        self.data_root = None
        self.store = None
        self.editor = None

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

        self.store = ProjectStore("editor-project")
        segments = _sample_segments()
        self.store.write_segments(segments)
        orders = {
            "line": ["p000_l0000"],
            "word": ["p000_l0000_w0000", "p000_l0000_w0001"],
        }
        segments_by_id = {segment.segment_id: segment for segment in segments}
        parents = {"p000_l0000_w0000": "p000_l0000", "p000_l0000_w0001": "p000_l0000"}
        edits: dict[str, str] = {}
        self.editor = SegmentEditor(orders, segments_by_id, parents, edits, self.store)

    def test_save_line_updates_words(self) -> None:
        assert self.editor is not None
        assert self.store is not None
        self.editor.save("p000_l0000", "line", "alpha beta")
        edits = self.store.read_edits()
        self.assertEqual(edits["p000_l0000"], "alpha beta")
        self.assertEqual(edits["p000_l0000_w0000"], "alpha")
        self.assertEqual(edits["p000_l0000_w0001"], "beta")
        master_text = self.store.master_path.read_text(encoding="utf-8")
        self.assertEqual(master_text.strip(), "alpha beta")

    def test_save_word_updates_parent(self) -> None:
        assert self.editor is not None
        assert self.store is not None
        self.editor.save("p000_l0000_w0000", "word", "HELLO")
        edits = self.store.read_edits()
        self.assertEqual(edits["p000_l0000_w0000"], "HELLO")
        self.assertTrue(edits["p000_l0000"].startswith("HELLO"))

    def test_get_text_prefers_edits(self) -> None:
        assert self.editor is not None
        self.editor.save("p000_l0000_w0001", "word", "PLANET")
        self.assertEqual(self.editor.get_text("p000_l0000_w0001"), "PLANET")

    def test_get_text_builds_line_from_words(self) -> None:
        assert self.editor is not None
        self.editor.save("p000_l0000_w0000", "word", "HELLO")
        self.editor.save("p000_l0000_w0001", "word", "THERE")
        self.assertEqual(self.editor.get_text("p000_l0000"), "HELLO THERE")

    def test_compose_master_text_includes_all_lines(self) -> None:
        assert self.editor is not None
        self.editor.save("p000_l0000", "line", "alpha beta")
        master_text = self.editor.compose_master_text()
        self.assertEqual(master_text, "alpha beta")
