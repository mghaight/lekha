from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import cast, override
from unittest.mock import patch
import warnings

from PIL import Image

from lekha.project import ProjectManifest, ProjectStore, Segment
from lekha.server import ProjectRuntime, create_app

warnings.simplefilter("ignore", ResourceWarning)


def _runtime_segments() -> list[Segment]:
    line_one = Segment(
        segment_id="p000_l0000",
        view="line",
        page_index=0,
        line_index=0,
        word_index=None,
        page_image="page.png",
        bbox={"x": 10, "y": 10, "w": 60, "h": 20},
        base_text="base one",
        consensus_text="one consensus",
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
        bbox={"x": 12, "y": 40, "w": 58, "h": 22},
        base_text="base two",
        consensus_text="two consensus",
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
        base_text="base",
        consensus_text="base",
        has_conflict=False,
    )
    word_one_b = Segment(
        segment_id="p000_l0000_w0001",
        view="word",
        page_index=0,
        line_index=0,
        word_index=1,
        page_image="page.png",
        bbox={"x": 32, "y": 12, "w": 18, "h": 15},
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
        bbox={"x": 15, "y": 40, "w": 22, "h": 16},
        base_text="two",
        consensus_text="two",
        has_conflict=True,
    )
    return [line_one, line_two, word_one_a, word_one_b, word_two_a]


class ProjectRuntimeTests(unittest.TestCase):
    temp_dir: tempfile.TemporaryDirectory[str] | None
    data_root: Path | None
    project_id: str
    store: ProjectStore | None
    segments: list[Segment]
    runtime: ProjectRuntime | None

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.temp_dir = None
        self.data_root = None
        self.project_id = ""
        self.store = None
        self.segments = []
        self.runtime = None

    @override
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_root = Path(self.temp_dir.name) / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.project_id = "runtime-project"
        patcher_project = patch("lekha.project.get_data_root", return_value=self.data_root)
        patcher_server = patch("lekha.server.get_data_root", return_value=self.data_root)
        _ = patcher_project.start()
        _ = patcher_server.start()
        self.addCleanup(patcher_project.stop)
        self.addCleanup(patcher_server.stop)

        self.store = ProjectStore(self.project_id)
        # Create an image for crop calculations.
        image_path = self.store.assets_dir / "page.png"
        Image.new("RGB", (200, 200), color="white").save(image_path)
        self.segments = list(_runtime_segments())
        self.store.write_segments(self.segments)
        self.store.write_manifest(
            ProjectManifest(
                project_id=self.project_id,
                source="source.pdf",
                languages=["eng"],
                models=["tesseract"],
                files=["page.png"],
            )
        )
        # Another manifest to exercise list_projects sorting.
        other_project_dir = self.data_root / "zeta-project"
        other_project_dir.mkdir(parents=True, exist_ok=True)
        _ = (other_project_dir / "manifest.json").write_text(
            '{"project_id": "zeta-project", "source": "zeta"}', encoding="utf-8"
        )
        self.runtime = ProjectRuntime(self.store)

    def test_ensure_state_defaults_to_first_line(self) -> None:
        assert self.runtime is not None
        state = self.runtime.ensure_state()
        self.assertEqual(state["view"], "line")
        self.assertEqual(state["segment_id"], "p000_l0000")

    def test_save_line_updates_words_and_master(self) -> None:
        assert self.runtime is not None
        assert self.store is not None
        self.runtime.save("p000_l0000", "line", "alpha beta")
        edits = self.store.read_edits()
        self.assertEqual(edits["p000_l0000"], "alpha beta")
        self.assertEqual(edits["p000_l0000_w0000"], "alpha")
        self.assertEqual(edits["p000_l0000_w0001"], "beta")
        master_text = self.store.master_path.read_text(encoding="utf-8")
        self.assertIn("alpha beta", master_text)

    def test_save_word_updates_parent_line(self) -> None:
        assert self.runtime is not None
        assert self.store is not None
        self.runtime.save("p000_l0000_w0000", "word", "HELLO")
        edits = self.store.read_edits()
        self.assertEqual(edits["p000_l0000_w0000"], "HELLO")
        self.assertTrue(edits["p000_l0000"].startswith("HELLO"))

    def test_navigation_and_switch_view(self) -> None:
        assert self.runtime is not None
        next_line = self.runtime.navigate("line", "p000_l0000", "next")
        self.assertEqual(next_line, "p000_l0001")
        same_line = self.runtime.navigate("line", "p000_l0001", "next")
        self.assertEqual(same_line, "p000_l0001")
        first_word = self.runtime.switch_view("p000_l0000", "word")
        self.assertEqual(first_word, "p000_l0000_w0000")
        back_to_line = self.runtime.switch_view(first_word, "line")
        self.assertEqual(back_to_line, "p000_l0000")

    def test_segment_payload_reflects_conflicts(self) -> None:
        assert self.runtime is not None
        payload = self.runtime.segment_payload("p000_l0001")
        self.assertTrue(payload["has_conflict"])
        self.runtime.save("p000_l0001", "line", "resolved")
        payload_after = self.runtime.segment_payload("p000_l0001")
        self.assertFalse(payload_after["has_conflict"])

    def test_crop_bounds_and_image_loading(self) -> None:
        assert self.runtime is not None
        bounds = self.runtime.get_crop_bounds("p000_l0000_w0000")
        self.assertGreater(bounds["width"], 0)
        self.assertGreater(bounds["height"], 0)
        segment = self.runtime.get_segment("p000_l0000_w0000")
        image = self.runtime.load_segment_image(segment, bounds)
        self.assertEqual(image.size, (bounds["width"], bounds["height"]))

    def test_create_app_endpoints(self) -> None:
        assert self.store is not None
        app = create_app(self.store)
        client = app.test_client()

        state_resp = client.get("/api/state")
        self.assertEqual(state_resp.status_code, 200)
        state_json = cast(dict[str, object], state_resp.get_json())
        self.assertEqual(state_json["project_id"], self.project_id)
        state_resp.close()

        seg_resp = client.get("/api/segment/p000_l0000")
        self.assertEqual(seg_resp.status_code, 200)
        payload = cast(dict[str, object], seg_resp.get_json())
        self.assertEqual(payload["segment_id"], "p000_l0000")
        seg_resp.close()

        image_resp = client.get("/api/segment/p000_l0000_w0000/image")
        self.assertEqual(image_resp.status_code, 200)
        self.assertEqual(image_resp.mimetype, "image/png")
        self.assertEqual(image_resp.headers["Cache-Control"], "no-store, no-cache, must-revalidate, max-age=0")
        image_resp.close()

        save_resp = client.post(
            "/api/save",
            json={
                "segment_id": "p000_l0000",
                "view": "line",
                "text": "gamma delta",
                "action": "save",
            },
        )
        self.assertEqual(save_resp.status_code, 200)
        save_json = cast(dict[str, object], save_resp.get_json())
        self.assertEqual(save_json["view"], "line")
        self.assertEqual(save_json["segment_id"], "p000_l0000")
        save_resp.close()

        view_resp = client.post("/api/view", json={"segment_id": "p000_l0000", "view": "word"})
        self.assertEqual(view_resp.status_code, 200)
        view_json = cast(dict[str, object], view_resp.get_json())
        self.assertEqual(view_json["view"], "word")
        view_resp.close()

        projects_resp = client.get("/api/projects")
        self.assertEqual(projects_resp.status_code, 200)
        projects_payload = cast(dict[str, object], projects_resp.get_json())
        projects = cast(Sequence[dict[str, object]], projects_payload["projects"])
        project_ids = {cast(str, item["project_id"]) for item in projects}
        self.assertIn(self.project_id, project_ids)
        self.assertIn("zeta-project", project_ids)
        projects_resp.close()

        project_resp = client.post("/api/project", json={"project_id": self.project_id})
        self.assertEqual(project_resp.status_code, 200)
        project_json = cast(dict[str, object], project_resp.get_json())
        self.assertEqual(project_json["project_id"], self.project_id)
        project_resp.close()

        export_resp = client.get("/api/export/master")
        self.assertEqual(export_resp.status_code, 200)
        self.assertEqual(export_resp.mimetype, "text/plain")
        export_resp.close()
