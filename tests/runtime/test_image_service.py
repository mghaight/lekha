"""Unit tests for `ImageService` image loading helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

from PIL import Image

from lekha.project import ProjectStore, Segment
from lekha.runtime.image_service import ImageService


def _segments() -> list[Segment]:
    line = Segment(
        segment_id="p000_l0000",
        view="line",
        page_index=0,
        line_index=0,
        word_index=None,
        page_image="page.png",
        bbox={"x": 5, "y": 5, "w": 50, "h": 20},
        base_text="line",
        consensus_text="line",
        has_conflict=False,
        word_ids=["p000_l0000_w0000"],
    )
    word = Segment(
        segment_id="p000_l0000_w0000",
        view="word",
        page_index=0,
        line_index=0,
        word_index=0,
        page_image="page.png",
        bbox={"x": 5, "y": 5, "w": 20, "h": 18},
        base_text="word",
        consensus_text="word",
        has_conflict=False,
    )
    return [line, word]


class ImageServiceTests(unittest.TestCase):
    """Direct unit tests for `ImageService` cropping and caching."""

    temp_dir: tempfile.TemporaryDirectory[str] | None
    data_root: Path | None
    store: ProjectStore | None
    service: ImageService | None

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.temp_dir = None
        self.data_root = None
        self.store = None
        self.service = None

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

        self.store = ProjectStore("image-project")
        segments = _segments()
        self.store.write_segments(segments)
        assert self.store.assets_dir.exists()
        # create image
        image_path = self.store.assets_dir / "page.png"
        Image.new("RGB", (200, 200), color="white").save(image_path)
        segments_by_id = {segment.segment_id: segment for segment in segments}
        self.service = ImageService(self.store, segments_by_id)

    def test_get_crop_bounds_caches_results(self) -> None:
        assert self.service is not None
        bounds_first = self.service.get_crop_bounds("p000_l0000_w0000")
        bounds_second = self.service.get_crop_bounds("p000_l0000_w0000")
        self.assertEqual(bounds_first, bounds_second)
        self.assertIn("p000_l0000_w0000", self.service.crop_cache)

    def test_load_segment_image_crops_region(self) -> None:
        assert self.service is not None
        segment = self.service.segments_by_id["p000_l0000_w0000"]
        image = self.service.load_segment_image(segment)
        self.assertGreater(image.width, 0)
        self.assertGreater(image.height, 0)

    def test_page_dimensions_cached(self) -> None:
        assert self.service is not None
        width, height = self.service._get_page_dimensions("page.png")  # pyright: ignore[reportPrivateUsage]
        self.assertEqual((width, height), (200, 200))
        width2, height2 = self.service._get_page_dimensions("page.png")  # pyright: ignore[reportPrivateUsage]
        self.assertEqual((width2, height2), (200, 200))

    def test_missing_image_raises(self) -> None:
        assert self.service is not None
        with self.assertRaises(FileNotFoundError):
            _ = self.service._get_page_dimensions("missing.png")  # pyright: ignore[reportPrivateUsage]

    def test_corrupted_image_raises(self) -> None:
        assert self.store is not None
        assert self.service is not None
        corrupted = self.store.assets_dir / "broken.png"
        _ = corrupted.write_text("not an image", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            _ = self.service._get_page_dimensions("broken.png")  # pyright: ignore[reportPrivateUsage]
