"""Integration with Tesseract OCR via pytesseract or CLI fallback."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import cast
from PIL import Image

try:
    import pytesseract
    from pytesseract import Output, TesseractError
except ImportError:  # pragma: no cover - surface at runtime
    pytesseract = None
    Output = None
    TesseractError = None


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
    words: list[TesseractWord]


@dataclass
class TesseractResult:
    text: str
    lines: list[TesseractLine]


def _language_arg(languages: Sequence[str]) -> str:
    return "+".join(languages) if languages else "eng"


def _safe_int(value: str | int, *, minimum: int = 0) -> int:
    """Convert pytesseract numeric fields to ints, with defensive fallbacks."""
    if isinstance(value, int):
        return max(value, minimum)
    try:
        return max(int(value), minimum)
    except (TypeError, ValueError):
        return minimum


def run_tesseract(image_path: Path, languages: Sequence[str]) -> TesseractResult:
    """Execute Tesseract OCR, preferring pytesseract for structured output."""
    if pytesseract is not None and Output is not None:
        image = Image.open(image_path)
        try:
            data = pytesseract.image_to_data(image, lang=_language_arg(languages), output_type=Output.DICT)
        except Exception as exc:  # pragma: no cover - defensive
            if TesseractError is not None and isinstance(exc, TesseractError):
                message = _language_hint_message(str(exc), languages)
                raise RuntimeError(message) from exc
            raise
        words: list[TesseractWord] = []
        lines: list[TesseractLine] = []
        current_line_id = None
        current_line_words: list[TesseractWord] = []
        current_bbox = None
        current_line_index = -1
        tokens: list[str] = []
        num_entries = len(data["text"])
        for idx in range(num_entries):
            text = data["text"][idx].strip()
            if not text:
                continue
            line_num = _safe_int(data["line_num"][idx])
            block_num = _safe_int(data["block_num"][idx])
            par_num = _safe_int(data["par_num"][idx])
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
                left=_safe_int(data["left"][idx]),
                top=_safe_int(data["top"][idx]),
                width=_safe_int(data["width"][idx], minimum=0),
                height=_safe_int(data["height"][idx], minimum=0),
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
        try:
            _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr_value = cast(str | None, exc.stderr)
            stdout_value = cast(str | None, exc.stdout)
            stderr = stderr_value or ""
            stdout = stdout_value or ""
            message = stderr or stdout or str(exc)
            hint = _language_hint_message(message, languages)
            raise RuntimeError(hint) from exc
        text = (output_base.with_suffix(".txt")).read_text(encoding="utf-8")
    # Without structured data, degrade gracefully to a single line.
    single_line = TesseractLine(text=text.replace("\n", " "), left=0, top=0, width=0, height=0, line_index=0, words=[])
    return TesseractResult(text=text, lines=[single_line])


def _language_hint_message(raw_message: str, languages: Sequence[str]) -> str:
    lang = _language_arg(languages)
    normalized = raw_message.lower()
    missing_patterns = [
        "error opening data file",
        "failed loading language",
        "couldn't load any languages",
        "could not initialize tesseract",
    ]
    if any(pattern in normalized for pattern in missing_patterns):
        return (
            f"Tesseract is missing the traineddata files required for language '{lang}'. "
            "Install the appropriate language data (update your tessdata directory or set TESSDATA_PREFIX) and retry."
        )
    return raw_message.strip() or "Tesseract OCR failed."
