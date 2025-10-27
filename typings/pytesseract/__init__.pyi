from __future__ import annotations

from typing import Mapping, TypedDict
from PIL import Image

class ImageDataDict(TypedDict):
    text: list[str]
    left: list[str]
    top: list[str]
    width: list[str]
    height: list[str]
    block_num: list[str]
    par_num: list[str]
    line_num: list[str]

class TesseractError(Exception): ...

class _OutputEnum:
    DICT: int

Output: _OutputEnum

def image_to_data(
    image: Image.Image,
    lang: str | None = ...,
    config: str = ...,
    nice: int = ...,
    output_type: int = ...,
    timeout: int = ...,
    pandas_config: Mapping[str, str] | None = ...,
) -> ImageDataDict: ...
