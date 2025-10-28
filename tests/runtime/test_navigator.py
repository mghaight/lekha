"""Unit tests for `SegmentNavigator` navigation helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from lekha.project import ProjectStore, Segment
from lekha.runtime.navigator import SegmentNavigator


def _build_segments() -> list[Segment]:
    line_one = Segment(
        segment_id="p000_l0000",
        view="line",
        page_index=0,
        line_index=0,
        word_index=None,
        page_image="page.png",
        bbox={"x": 10, "y": 10, "w": 60, "h": 20},
        base_text="line one",
        consensus_text="line one",
        has_conflict=False,
        word_ids=["p000_l0000_w0000", "p000_l0000_w0001"],
    )
    line_two = Segment(
        segment_id="p000_l0001",
        view="line",
        page_index=0,
        line_index=1,
        word_index=None,
        page_image="page.png",
        bbox={"x": 10, "y": 40, "w": 60, "h": 20},
        base_text="line two",
        consensus_text="line two",
        has_conflict=True,
        word_ids=["p000_l0001_w0000"],
    )
    word_one_a = Segment(
        segment_id="p000_l0000_w0000",
        view="word",
        page_index=0,
        line_index=0,
        word_index=0,
        page_image="page.png",
        bbox={"x": 10, "y": 10, "w": 20, "h": 18},
        base_text="line",
        consensus_text="line",
        has_conflict=False,
    )
    word_one_b = Segment(
        segment_id="p000_l0000_w0001",
        view="word",
        page_index=0,
        line_index=0,
        word_index=1,
        page_image="page.png",
        bbox={"x": 35, "y": 10, "w": 20, "h": 18},
        base_text="one",
        consensus_text="one",
        has_conflict=False,
    )
    word_two_a = Segment(
        segment_id="p000_l0001_w0000",
        view="word",
        page_index=0,
        line_index=1,
        word_index=0,
        page_image="page.png",
        bbox={"x": 10, "y": 40, "w": 20, "h": 18},
        base_text="two",
        consensus_text="two",
        has_conflict=True,
    )
    return [line_one, line_two, word_one_a, word_one_b, word_two_a]


class SegmentNavigatorTests(unittest.TestCase):
    """Direct unit tests for `SegmentNavigator` traversal helpers."""

    temp_dir: tempfile.TemporaryDirectory[str] | None
    data_root: Path | None
    store: ProjectStore | None
    navigator: SegmentNavigator | None

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.temp_dir = None
        self.data_root = None
        self.store = None
        self.navigator = None

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

        self.store = ProjectStore("navigator-project")
        segments = _build_segments()
        self.store.write_segments(segments)
        self.store.write_state({"view": "line", "segment_id": "p000_l0000"})
        orders = {
            "line": ["p000_l0000", "p000_l0001"],
            "word": ["p000_l0000_w0000", "p000_l0000_w0001", "p000_l0001_w0000"],
        }
        segments_by_id = {segment.segment_id: segment for segment in segments}
        parents = {
            "p000_l0000_w0000": "p000_l0000",
            "p000_l0000_w0001": "p000_l0000",
            "p000_l0001_w0000": "p000_l0001",
        }
        edits: dict[str, str] = {}
        state = {"view": "line", "segment_id": "p000_l0000"}
        self.navigator = SegmentNavigator(orders, segments_by_id, parents, edits, state, self.store)

    def test_navigate_next_prev(self) -> None:
        assert self.navigator is not None
        next_line = self.navigator.navigate("line", "p000_l0000", "next")
        self.assertEqual(next_line, "p000_l0001")
        prev_line = self.navigator.navigate("line", "p000_l0001", "prev")
        self.assertEqual(prev_line, "p000_l0000")

    def test_next_issue_finds_conflict(self) -> None:
        assert self.navigator is not None
        issue = self.navigator.navigate("line", "p000_l0000", "next_issue")
        self.assertEqual(issue, "p000_l0001")
        # mark as resolved
        self.navigator.edits["p000_l0001"] = "resolved"
        self.assertEqual(self.navigator.navigate("line", "p000_l0000", "next_issue"), "p000_l0000")

    def test_switch_view_preserves_context(self) -> None:
        assert self.navigator is not None
        to_word = self.navigator.switch_view("p000_l0000", "word")
        self.assertEqual(to_word, "p000_l0000_w0000")
        back_to_line = self.navigator.switch_view(to_word, "line")
        self.assertEqual(back_to_line, "p000_l0000")

    def test_switch_view_fallbacks_when_no_children(self) -> None:
        assert self.navigator is not None
        # remove children to trigger fallback
        self.navigator.orders["word"] = []
        target = self.navigator.switch_view("unknown", "word")
        self.assertEqual(target, "unknown")

    def test_persist_state_writes_to_store(self) -> None:
        assert self.navigator is not None
        assert self.store is not None
        self.navigator.persist_state("word", "p000_l0000_w0001")
        state = self.store.read_state()
        self.assertEqual(state["view"], "word")
        self.assertEqual(state["segment_id"], "p000_l0000_w0001")

    def test_navigation_status_flags(self) -> None:
        assert self.navigator is not None
        status = self.navigator.navigation_status("p000_l0000", "line")
        self.assertEqual(status["can_prev"], False)
        self.assertEqual(status["can_next"], True)
        self.assertEqual(status["has_next_issue"], True)
