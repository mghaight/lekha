from __future__ import annotations

import os
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import cast, override
from unittest.mock import patch
import warnings

from PIL import Image

from lekha.project import ProjectManifest, ProjectStore, Segment
from lekha.server import ProjectRuntime, create_app, get_or_generate_secret_key

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

    def test_get_page_dimensions_lazy_loads_and_caches(self) -> None:
        assert self.runtime is not None
        # Dimensions not loaded yet
        self.assertNotIn("page.png", self.runtime.page_dimensions)
        # First call loads dimensions (testing private method is acceptable in tests)
        width, height = self.runtime._get_page_dimensions("page.png")  # pyright: ignore[reportPrivateUsage]
        self.assertEqual(width, 200)
        self.assertEqual(height, 200)
        # Now cached
        self.assertIn("page.png", self.runtime.page_dimensions)
        # Second call uses cache
        width2, height2 = self.runtime._get_page_dimensions("page.png")  # pyright: ignore[reportPrivateUsage]
        self.assertEqual(width2, 200)
        self.assertEqual(height2, 200)

    def test_get_page_dimensions_raises_on_missing_file(self) -> None:
        assert self.runtime is not None
        with self.assertRaises(FileNotFoundError) as ctx:
            _ = self.runtime._get_page_dimensions("nonexistent.png")  # pyright: ignore[reportPrivateUsage]
        self.assertIn("Image file not found", str(ctx.exception))

    def test_get_page_dimensions_raises_on_corrupted_file(self) -> None:
        assert self.runtime is not None
        assert self.store is not None
        # Create a corrupted image file
        corrupted_path = self.store.assets_dir / "corrupted.png"
        _ = corrupted_path.write_text("not an image", encoding="utf-8")
        with self.assertRaises(RuntimeError) as ctx:
            _ = self.runtime._get_page_dimensions("corrupted.png")  # pyright: ignore[reportPrivateUsage]
        self.assertIn("Failed to load image dimensions", str(ctx.exception))

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

    def test_missing_image_file_returns_404(self) -> None:
        assert self.store is not None
        assert self.runtime is not None
        # Delete the image file to simulate missing file
        image_path = self.store.assets_dir / "page.png"
        image_path.unlink()

        app = create_app(self.store)
        client = app.test_client()

        image_resp = client.get("/api/segment/p000_l0000_w0000/image")
        self.assertEqual(image_resp.status_code, 404)
        self.assertIn("Image file not found", image_resp.get_data(as_text=True))
        image_resp.close()

    def test_corrupted_image_file_returns_500(self) -> None:
        assert self.store is not None
        # Corrupt the image file by writing invalid data
        image_path = self.store.assets_dir / "page.png"
        _ = image_path.write_text("not a valid image", encoding="utf-8")

        app = create_app(self.store)
        client = app.test_client()

        image_resp = client.get("/api/segment/p000_l0000_w0000/image")
        self.assertEqual(image_resp.status_code, 500)
        response_text = image_resp.get_data(as_text=True)
        # Should have a meaningful error message (could be from dimension loading or image loading)
        self.assertTrue(
            "Failed to load image dimensions" in response_text or "Failed to load or process image" in response_text,
            f"Expected error message not found in response: {response_text}",
        )
        image_resp.close()


class SecretKeyTests(unittest.TestCase):
    def test_get_or_generate_secret_key_with_custom_env(self) -> None:
        with patch.dict(os.environ, {"LEKHA_WEB_SECRET": "custom-secret-key"}, clear=False):
            key = get_or_generate_secret_key()
            self.assertEqual(key, "custom-secret-key")

    def test_get_or_generate_secret_key_warns_for_default(self) -> None:
        with patch.dict(os.environ, {"LEKHA_WEB_SECRET": "lekha-dev"}, clear=False), patch(
            "lekha.server.logger"
        ) as mock_logger:
            key = get_or_generate_secret_key()
            self.assertEqual(key, "lekha-dev")
            # Mock object methods are dynamically typed, so type checking is suppressed
            mock_logger.warning.assert_called_once()  # pyright: ignore[reportAny]
            call_args = cast(tuple[tuple[object, ...], dict[str, object]], mock_logger.warning.call_args)  # pyright: ignore[reportAny]
            args, _ = call_args
            self.assertIn("default secret key", str(args[0]))

    def test_get_or_generate_secret_key_generates_if_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            key = get_or_generate_secret_key()
            # Should be a hex string of length 64 (32 bytes as hex)
            self.assertEqual(len(key), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in key))

    def test_generated_keys_are_unique(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            key1 = get_or_generate_secret_key()
            key2 = get_or_generate_secret_key()
            self.assertNotEqual(key1, key2)
