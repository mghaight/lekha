from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from lekha.project import (
    ProjectManifest,
    ProjectStore,
    Segment,
    project_id_for_path,
    slugify,
)


def _sample_segments() -> list[Segment]:
    line = Segment(
        segment_id="p000_l0000",
        view="line",
        page_index=0,
        line_index=0,
        word_index=None,
        page_image="page.png",
        bbox={"x": 5, "y": 10, "w": 40, "h": 12},
        base_text="base line",
        consensus_text="consensus line",
        has_conflict=False,
        alternatives={"alt": "value"},
        word_ids=["p000_l0000_w0000", "p000_l0000_w0001"],
    )
    word_one = Segment(
        segment_id="p000_l0000_w0000",
        view="word",
        page_index=0,
        line_index=0,
        word_index=0,
        page_image="page.png",
        bbox={"x": 5, "y": 10, "w": 12, "h": 10},
        base_text="base",
        consensus_text="consensus1",
        has_conflict=False,
    )
    word_two = Segment(
        segment_id="p000_l0000_w0001",
        view="word",
        page_index=0,
        line_index=0,
        word_index=1,
        page_image="page.png",
        bbox={"x": 20, "y": 10, "w": 18, "h": 10},
        base_text="text",
        consensus_text="consensus2",
        has_conflict=True,
    )
    return [line, word_one, word_two]


class ProjectUtilitiesTests(unittest.TestCase):
    def test_slugify_removes_invalid_characters(self) -> None:
        self.assertEqual(slugify("Hello, World!"), "hello-world")
        self.assertEqual(slugify("Already-clean"), "already-clean")
        self.assertEqual(slugify("123 456"), "123-456")

    def test_project_id_for_path_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "Some Project"
            path.mkdir()
            first = project_id_for_path(path)
            second = project_id_for_path(path)
        self.assertEqual(first, second)
        self.assertIn("some-project", first)


class ProjectStoreTests(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    data_root: Path | None = None

    @override
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_root = Path(self.temp_dir.name) / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)
        patcher = patch("lekha.project.get_data_root", return_value=self.data_root)
        self.addCleanup(patcher.stop)
        _ = patcher.start()

    def test_manifest_round_trip(self) -> None:
        store = ProjectStore("project-1")
        manifest = ProjectManifest(
            project_id="project-1",
            source="source.pdf",
            languages=["eng", "san"],
            models=["tesseract"],
            files=["page1.png"],
        )
        store.write_manifest(manifest)
        loaded = store.load_manifest()
        self.assertEqual(loaded, manifest)

    def test_segments_round_trip(self) -> None:
        store = ProjectStore("project-2")
        segments = list(_sample_segments())
        store.write_segments(segments)
        loaded = store.load_segments()
        self.assertEqual(loaded, segments)

    def test_read_edits_filters_non_strings(self) -> None:
        store = ProjectStore("project-3")
        _ = store.edits_path.write_text(json.dumps({"keep": "value", "skip": 123}), encoding="utf-8")
        edits = store.read_edits()
        self.assertEqual(edits, {"keep": "value"})

    def test_read_state_adds_defaults(self) -> None:
        store = ProjectStore("project-4")
        store.write_state({})
        state = store.read_state()
        self.assertEqual(state["view"], "line")
        self.assertEqual(state["segment_id"], "")

    def test_write_master_creates_file(self) -> None:
        store = ProjectStore("project-5")
        store.write_master("hello world")
        self.assertTrue(store.master_path.exists())
        self.assertEqual(store.master_path.read_text(encoding="utf-8"), "hello world")
