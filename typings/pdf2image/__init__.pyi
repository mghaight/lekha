from __future__ import annotations

from collections.abc import Mapping
from PIL import Image

def convert_from_path(
    pdf_path: str,
    dpi: int = ...,
    output_folder: str | None = ...,
    first_page: int | None = ...,
    last_page: int | None = ...,
    fmt: str = ...,
    jpegopt: Mapping[str, object] | None = ...,
    thread_count: int = ...,
    userpw: str | None = ...,
    ownerpw: str | None = ...,
    use_cropbox: bool = ...,
    strict: bool = ...,
    transparent: bool = ...,
    single_file: bool = ...,
    output_file: str | None = ...,
    poppler_path: str | None = ...,
    grayscale: bool = ...,
    size: tuple[int, int] | int | None = ...,
    paths_only: bool = ...,
    use_pdftocairo: bool = ...,
    timeout: int | None = ...,
    hide_annotations: bool = ...,
) -> list[Image.Image]: ...
