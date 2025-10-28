"""Service for loading and processing segment images."""

from __future__ import annotations

from PIL import Image

from ..project import ProjectStore, Segment


class ImageService:
    """Handles image loading, cropping, and dimension caching for segments."""

    def __init__(self, store: ProjectStore, segments_by_id: dict[str, Segment]):
        self.store: ProjectStore = store
        self.segments_by_id: dict[str, Segment] = segments_by_id
        self.page_dimensions: dict[str, tuple[int, int]] = {}
        self.crop_cache: dict[str, dict[str, int]] = {}

    def load_segment_image(self, segment: Segment, crop: dict[str, int] | None = None) -> Image.Image:
        """
        Load and crop a segment image.

        Args:
            segment: The segment to load an image for
            crop: Optional crop bounds dict, otherwise computed from segment

        Returns:
            Cropped PIL Image

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If image cannot be loaded or processed
        """
        image_path = self.store.assets_dir / segment.page_image
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {segment.page_image}")
        crop_bounds = crop or self.get_crop_bounds(segment.segment_id)
        try:
            with Image.open(image_path) as image:
                left = crop_bounds["left"]
                top = crop_bounds["top"]
                right = crop_bounds["right"]
                bottom = crop_bounds["bottom"]
                return image.crop((left, top, right, bottom)).copy()
        except (OSError, IOError) as exc:
            raise RuntimeError(f"Failed to load or process image {segment.page_image}: {exc}") from exc

    def get_crop_bounds(self, segment_id: str) -> dict[str, int]:
        """
        Get crop bounds for a segment, using cache if available.

        Args:
            segment_id: ID of the segment

        Returns:
            Dictionary with keys: left, top, right, bottom, width, height
        """
        if segment_id not in self.crop_cache:
            segment = self.segments_by_id[segment_id]
            crop = self._crop_geometry(segment)
            self.crop_cache[segment_id] = crop
        return self.crop_cache[segment_id]

    def _crop_geometry(
        self, segment: Segment, padding_x_ratio: float = 0.1, padding_y_ratio: float = 0.5
    ) -> dict[str, int]:
        """
        Calculate crop geometry for a segment with padding.

        Args:
            segment: The segment to calculate crop for
            padding_x_ratio: Horizontal padding as ratio of segment width
            padding_y_ratio: Vertical padding as ratio of segment height

        Returns:
            Dictionary with crop bounds and dimensions
        """
        width, height = self._get_page_dimensions(segment.page_image)
        bbox = segment.bbox or {}
        base_x = bbox.get("x", 0)
        base_y = bbox.get("y", 0)
        base_w = max(bbox.get("w", 1), 1)
        base_h = max(bbox.get("h", 1), 1)
        pad_x = max(int(base_w * padding_x_ratio), 2)
        pad_y = max(int(base_h * padding_y_ratio), 2)
        left = max(base_x - pad_x, 0)
        top = max(base_y - pad_y, 0)
        right = min(base_x + base_w + pad_x, width)
        bottom = min(base_y + base_h + pad_y, height)
        crop_width = max(right - left, 1)
        crop_height = max(bottom - top, 1)
        return {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": crop_width,
            "height": crop_height,
        }

    def _get_page_dimensions(self, page_image: str) -> tuple[int, int]:
        """
        Lazily load page dimensions for an image.
        Dimensions are cached after first load.

        Args:
            page_image: Name of the page image file

        Returns:
            Tuple of (width, height)

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If image cannot be loaded
        """
        if page_image in self.page_dimensions:
            return self.page_dimensions[page_image]

        image_path = self.store.assets_dir / page_image
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {page_image}")

        try:
            with Image.open(image_path) as image:
                dimensions = image.size
                self.page_dimensions[page_image] = dimensions
                return dimensions
        except (OSError, IOError) as exc:
            raise RuntimeError(f"Failed to load image dimensions for {page_image}: {exc}") from exc
