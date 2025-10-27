from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections.abc import Sequence
from typing import cast, override

from typer.testing import CliRunner

from lekha.cli import app


class FakeStore:
    project_id: str

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id


class CLITests(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.runner: CliRunner = CliRunner()

    @override
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_cli_launches_existing_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            project_dir = data_root / "proj-one"
            project_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "project_id": "proj-one",
                "source": "source.pdf",
                "languages": ["eng"],
                "models": ["tesseract"],
                "files": ["page.png"],
            }
            _ = (project_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            fake_app = object()
            run_server_mock = MagicMock()

            with patch("lekha.cli.get_data_root", return_value=data_root), patch(
                "lekha.project.get_data_root", return_value=data_root
            ), patch("lekha.cli.ProjectStore", side_effect=FakeStore) as store_mock, patch(
                "lekha.cli.create_app", return_value=fake_app
            ) as create_app_mock, patch(
                "lekha.cli.run_server", new=run_server_mock
            ), patch(
                "lekha.cli.webbrowser.open"
            ) as browser_mock:
                result = self.runner.invoke(app, [], input="1\n")

        self.assertEqual(result.exit_code, 0)
        store_mock.assert_called_once_with("proj-one")
        create_app_mock.assert_called_once()
        run_server_mock.assert_called_once_with(fake_app, port=8765)
        browser_mock.assert_called_once()

    def test_cli_processes_new_manuscript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir) / "store"
            manuscript_dir = Path(tmp_dir) / "manuscript"
            manuscript_dir.mkdir()
            _ = (manuscript_dir / "page1.png").write_text("fake image data", encoding="utf-8")

            run_server_mock = MagicMock()

            with patch("lekha.cli.get_data_root", return_value=data_root), patch(
                "lekha.project.get_data_root", return_value=data_root
            ), patch("lekha.cli.ProjectStore", side_effect=FakeStore) as store_mock, patch(
                "lekha.cli.project_id_for_path", return_value="proj-new"
            ), patch("lekha.cli.create_app", return_value=object()) as create_app_mock, patch(
                "lekha.cli.run_server", new=run_server_mock
            ), patch(
                "lekha.cli.webbrowser.open"
            ) as browser_mock, patch(
                "lekha.cli.process_inputs"
            ) as process_mock:
                result = self.runner.invoke(app, [str(manuscript_dir)])

        self.assertEqual(result.exit_code, 0)
        store_mock.assert_called_with("proj-new")
        process_mock.assert_called_once()
        call_args = cast(tuple[tuple[object, ...], dict[str, object]], process_mock.call_args)
        args_tuple, kwargs_dict = call_args
        source_paths = list(cast(Sequence[Path], args_tuple[0]))
        self.assertEqual([path.name for path in source_paths], ["page1.png"])
        self.assertEqual(cast(list[str], kwargs_dict["languages"]), ["eng"])
        self.assertEqual(cast(list[str], kwargs_dict["models"]), ["tesseract"])
        store_arg = cast(FakeStore, kwargs_dict["store"])
        self.assertEqual(store_arg.project_id, "proj-new")
        browser_mock.assert_called_once()
        create_app_mock.assert_called_once()
        run_server_mock.assert_called_once()

    def test_cli_respects_custom_languages_and_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            manuscript_path = Path(tmp_dir) / "document.pdf"
            _ = manuscript_path.write_text("PDF placeholder", encoding="utf-8")

            run_server_mock = MagicMock()

            with patch("lekha.cli.get_data_root", return_value=data_root), patch(
                "lekha.project.get_data_root", return_value=data_root
            ), patch("lekha.cli.ProjectStore", side_effect=FakeStore), patch(
                "lekha.cli.project_id_for_path", return_value="proj-custom"
            ), patch("lekha.cli.create_app", return_value=object()), patch(
                "lekha.cli.run_server", new=run_server_mock
            ), patch(
                "lekha.cli.webbrowser.open"
            ) as browser_mock, patch(
                "lekha.cli.process_inputs"
            ) as process_mock:
                result = self.runner.invoke(
                    app,
                    [
                        "--lang",
                        "san",
                        "--lang",
                        "pra",
                        "--model",
                        "custom",
                        "--no-browser",
                        "--port",
                        "9999",
                        str(manuscript_path),
                    ],
                )

        self.assertEqual(result.exit_code, 0)
        call_args = cast(tuple[tuple[object, ...], dict[str, object]], process_mock.call_args)
        _, kwargs_dict = call_args
        languages = cast(list[str], kwargs_dict["languages"])
        models = cast(list[str], kwargs_dict["models"])
        self.assertEqual(languages, ["san", "pra"])
        self.assertEqual(models, ["custom"])
        browser_mock.assert_not_called()
        run_server_mock.assert_called_once()
        port = cast(int, cast(dict[str, object], run_server_mock.call_args.kwargs)["port"])
        self.assertEqual(port, 9999)
