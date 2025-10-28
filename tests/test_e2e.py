"""Deterministic end-to-end workflow tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from PIL import Image
from typer.testing import CliRunner

from lekha.cli import app as cli_app
from lekha.ocr.tesseract_engine import TesseractLine, TesseractResult, TesseractWord
from lekha.project import ProjectStore
from lekha.server import ProjectRuntime


class EndToEndWorkflowTests(unittest.TestCase):
    """Exercise CLI → processing → API flow with stubbed OCR output."""

    temp_dir: tempfile.TemporaryDirectory[str] | None
    data_root: Path | None
    runner: CliRunner

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.temp_dir = None
        self.data_root = None
        self.runner = CliRunner()

    @override
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        base_path = Path(self.temp_dir.name)
        self.data_root = base_path / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)

        patchers = [
            patch("lekha.project.get_data_root", return_value=self.data_root),
            patch("lekha.cli.get_data_root", return_value=self.data_root),
            patch("lekha.server.get_data_root", return_value=self.data_root),
        ]
        for patcher in patchers:
            self.addCleanup(patcher.stop)
            patcher.start()

        self.runner = CliRunner()

    def _create_fixture_image(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / "sample.png"
        Image.new("RGB", (160, 60), color="white").save(image_path)
        return image_path

    def test_cli_to_api_round_trip(self) -> None:
        assert self.temp_dir is not None
        assert self.data_root is not None

        manuscript_dir = Path(self.temp_dir.name) / "manuscript"
        image_path = self._create_fixture_image(manuscript_dir)

        word_hello = TesseractWord(
            text="Hello",
            left=10,
            top=10,
            width=40,
            height=18,
            line_index=0,
            word_index=0,
        )
        word_world = TesseractWord(
            text="World",
            left=60,
            top=10,
            width=42,
            height=18,
            line_index=0,
            word_index=1,
        )
        line = TesseractLine(
            text="Hello World",
            left=8,
            top=8,
            width=100,
            height=26,
            line_index=0,
            words=[word_hello, word_world],
        )
        tess_result = TesseractResult(text="Hello World", lines=[line])

        def fake_run_tesseract(image: Path, languages: list[str]) -> TesseractResult:
            self.assertTrue(image.exists())
            self.assertIn("eng", languages)
            return tess_result

        with (
            patch("lekha.cli.project_id_for_path", return_value="proj-e2e"),
            patch("lekha.processing.validate_tesseract_installation"),
            patch("lekha.processing._run_tesseract_with_logging", side_effect=fake_run_tesseract),
            patch("lekha.cli.webbrowser.open"),
            patch("lekha.cli.run_server") as run_server_mock,
        ):
            result = self.runner.invoke(
                cli_app,
                ["--no-browser", str(manuscript_dir)],
                catch_exceptions=False,
            )

        self.assertEqual(result.exit_code, 0, msg=f"CLI failed: {result.stdout}")
        run_server_mock.assert_called_once()

        store = ProjectStore("proj-e2e")
        manifest = store.load_manifest()
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest.project_id, "proj-e2e")
        self.assertEqual(len(manifest.files), 1)
        manifest_path = Path(manifest.files[0])
        self.assertEqual(manifest_path.resolve(), image_path.resolve())

        segments = store.load_segments()
        self.assertEqual(len(segments), 3)
        segment_ids = {segment.segment_id for segment in segments}
        self.assertIn("p000_l0000", segment_ids)
        self.assertIn("p000_l0000_w0000", segment_ids)
        self.assertIn("p000_l0000_w0001", segment_ids)

        line_segment = next(segment for segment in segments if segment.segment_id == "p000_l0000")
        self.assertEqual(line_segment.consensus_text, "Hello World")
        self.assertEqual(line_segment.word_ids, ["p000_l0000_w0000", "p000_l0000_w0001"])

        word_segment = next(segment for segment in segments if segment.segment_id == "p000_l0000_w0000")
        self.assertEqual(word_segment.base_text, "Hello")
        self.assertEqual(word_segment.consensus_text, "Hello")

        master_text = store.master_path.read_text(encoding="utf-8")
        self.assertEqual(master_text.strip(), "Hello World")

        runtime = ProjectRuntime(store)
        self.assertEqual(runtime.get_text("p000_l0000"), "Hello World")
        self.assertEqual(runtime.get_text("p000_l0000_w0001"), "World")

        app_instance = run_server_mock.call_args.args[0]
        with app_instance.test_client() as client:
            state_resp = client.get("/api/state")
            self.assertEqual(state_resp.status_code, 200)
            state_payload = state_resp.get_json()
            assert isinstance(state_payload, dict)
            self.assertEqual(state_payload["project_id"], "proj-e2e")
            self.assertEqual(state_payload["segment_id"], "p000_l0000")

            segment_resp = client.get("/api/segment/p000_l0000")
            self.assertEqual(segment_resp.status_code, 200)
            payload = segment_resp.get_json()
            assert isinstance(payload, dict)
            self.assertEqual(payload["text"], "Hello World")

            image_resp = client.get("/api/segment/p000_l0000_w0000/image")
            self.assertEqual(image_resp.status_code, 200)
            self.assertEqual(image_resp.mimetype, "image/png")
            cache_header = image_resp.headers.get("Cache-Control")
            self.assertIn("no-store", cache_header or "")

            projects_resp = client.get("/api/projects")
            self.assertEqual(projects_resp.status_code, 200)
            projects_payload = projects_resp.get_json()
            assert isinstance(projects_payload, dict)
            projects_json = projects_payload.get("projects")
            assert isinstance(projects_json, list)
            project_ids = {item["project_id"] for item in projects_json if isinstance(item, dict)}
            self.assertIn("proj-e2e", project_ids)

            export_resp = client.get("/api/export/master")
            self.assertEqual(export_resp.status_code, 200)
            self.assertEqual(export_resp.mimetype, "text/plain")
            export_text = export_resp.get_data(as_text=True)
            self.assertEqual(export_text.strip(), "Hello World")

        outputs_path = store.outputs_dir / "tesseract" / "page_0000.txt"
        self.assertTrue(outputs_path.exists())
        self.assertEqual(outputs_path.read_text(encoding="utf-8").strip(), "Hello World")

        state_contents = store.read_state()
        self.assertEqual(state_contents["view"], "line")
        self.assertEqual(state_contents["segment_id"], "p000_l0000")
