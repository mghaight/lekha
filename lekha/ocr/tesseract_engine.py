"""Integration with Tesseract OCR via pytesseract or CLI fallback."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from PIL import Image

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover - surface at runtime
    pytesseract = None
    Output = None


@dataclass
class TesseractWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    line_index: int
    word_index: int


@dataclass
class TesseractLine:
    text: str
    left: int
    top: int
    width: int
    height: int
    line_index: int
    words: List[TesseractWord]


@dataclass
class TesseractResult:
    text: str
    lines: List[TesseractLine]


def _language_arg(languages: Sequence[str]) -> str:
    return "+".join(languages) if languages else "eng"


def run_tesseract(image_path: Path, languages: Sequence[str]) -> TesseractResult:
    """Execute Tesseract OCR, preferring pytesseract for structured output."""
    if pytesseract is not None and Output is not None:
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, lang=_language_arg(languages), output_type=Output.DICT)
        words: List[TesseractWord] = []
        lines: List[TesseractLine] = []
        current_line_id = None
        current_line_words: List[TesseractWord] = []
        current_bbox = None
        current_line_index = -1
        tokens: List[str] = []
        num_entries = len(data["text"])
        for idx in range(num_entries):
            text = data["text"][idx].strip()
            if not text:
                continue
            line_num = data["line_num"][idx]
            block_num = data["block_num"][idx]
            par_num = data["par_num"][idx]
            compound_index = (block_num, par_num, line_num)
            if current_line_id != compound_index:
                if current_line_words:
                    lefts = [w.left for w in current_line_words]
                    tops = [w.top for w in current_line_words]
                    rights = [w.left + w.width for w in current_line_words]
                    bottoms = [w.top + w.height for w in current_line_words]
                    current_bbox = (
                        min(lefts),
                        min(tops),
                        max(rights) - min(lefts),
                        max(bottoms) - min(tops),
                    )
                    lines.append(
                        TesseractLine(
                            text=" ".join(word.text for word in current_line_words),
                            left=current_bbox[0],
                            top=current_bbox[1],
                            width=current_bbox[2],
                            height=current_bbox[3],
                            line_index=current_line_index,
                            words=current_line_words,
                        )
                    )
                current_line_words = []
                current_line_id = compound_index
                current_line_index += 1
            word = TesseractWord(
                text=text,
                left=data["left"][idx],
                top=data["top"][idx],
                width=data["width"][idx],
                height=data["height"][idx],
                line_index=current_line_index,
                word_index=len(current_line_words),
            )
            current_line_words.append(word)
            words.append(word)
            tokens.append(text)

        if current_line_words:
            lefts = [w.left for w in current_line_words]
            tops = [w.top for w in current_line_words]
            rights = [w.left + w.width for w in current_line_words]
            bottoms = [w.top + w.height for w in current_line_words]
            current_bbox = (
                min(lefts),
                min(tops),
                max(rights) - min(lefts),
                max(bottoms) - min(tops),
            )
            lines.append(
                TesseractLine(
                    text=" ".join(word.text for word in current_line_words),
                    left=current_bbox[0],
                    top=current_bbox[1],
                    width=current_bbox[2],
                    height=current_bbox[3],
                    line_index=current_line_index,
                    words=current_line_words,
                )
            )

        return TesseractResult(text="\n".join(line.text for line in lines), lines=lines)

    # Fallback: shell out to `tesseract` CLI and parse plain text.
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir) / "output"
        cmd = [
            "tesseract",
            str(image_path),
            str(output_base),
            "-l",
            _language_arg(languages),
        ]
        subprocess.run(cmd, check=True)
        text = (output_base.with_suffix(".txt")).read_text(encoding="utf-8")
    # Without structured data, degrade gracefully to a single line.
    single_line = TesseractLine(text=text.replace("\n", " "), left=0, top=0, width=0, height=0, line_index=0, words=[])
    return TesseractResult(text=text, lines=[single_line])
