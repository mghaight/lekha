from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lekha.config import APP_NAME, get_data_root


class GetDataRootTests(unittest.TestCase):
    def test_posix_default_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            home_path = base / "home"
            with patch("platform.system", return_value="Linux"), patch(
                "pathlib.Path.home", return_value=home_path
            ), patch.dict(os.environ, {}, clear=True):
                root = get_data_root()
                expected = home_path / ".local" / "share" / APP_NAME
                self.assertEqual(root, expected)
                self.assertTrue(root.exists())

    def test_posix_respects_xdg_data_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            xdg_path = base / "xdg-data"
            with patch("platform.system", return_value="Darwin"), patch(
                "pathlib.Path.home", return_value=base / "should_not_be_used"
            ), patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg_path)}, clear=True):
                root = get_data_root()
                self.assertEqual(root, xdg_path / APP_NAME)
                self.assertTrue(root.exists())

    def test_windows_uses_localappdata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            local_app_data = base / "LocalAppData"
            with patch("platform.system", return_value="Windows"), patch(
                "pathlib.Path.home", return_value=base / "home"
            ), patch.dict(os.environ, {"LOCALAPPDATA": str(local_app_data)}, clear=True):
                root = get_data_root()
                self.assertEqual(root, local_app_data / APP_NAME)
                self.assertTrue(root.exists())
