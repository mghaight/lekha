"""High-level orchestration for OCR processing and diff generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping

from PIL import Image

from .diffing import BaseToken, WordConsensus, compute_word_consensus
from .ocr import TesseractResult, run_tesseract
from .ocr.tesseract_engine import validate_tesseract_installation
from .project import ProjectManifest, ProjectStore, Segment

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {"tesseract"}


@dataclass
class NormalizedLine:
    line_index: int
    text: str
    bbox: dict[str, int]
    words: list["NormalizedWord"]


@dataclass
class NormalizedWord:
    text: str
    bbox: dict[str, int]
    line_index: int
    word_index: int


def process_inputs(
    source_paths: Iterable[Path],
    languages: list[str],
    models: list[str],
    store: ProjectStore,
    source: str,
) -> None:
    """Process the provided sources with OCR models and persist results."""
    # Validate Tesseract installation before starting
    validate_tesseract_installation()

    languages = languages or ["eng"]
    selected_models = list(dict.fromkeys(models or ["tesseract"]))
    for model in selected_models:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported OCR model: {model}")
    if "tesseract" not in selected_models:
        selected_models.insert(0, "tesseract")

    source_paths = list(source_paths)
    manifest = ProjectManifest(
        project_id=store.project_id,
        source=source,
        languages=languages,
        models=selected_models,
        files=[str(path) for path in source_paths],
    )
    store.write_manifest(manifest)

    page_images = _prepare_page_images(source_paths, store)

    segments: list[Segment] = []
    master_lines: dict[tuple[int, int], str] = {}

    for page_index, image_path in enumerate(page_images):
        absolute_image_path = store.assets_dir / image_path
        with Image.open(absolute_image_path) as image:
            width, height = image.size

        tess_result = _run_tesseract_with_logging(absolute_image_path, languages)
        other_outputs: dict[str, str] = {"tesseract": tess_result.text}

        _persist_model_outputs(store, page_index, other_outputs)

        normalized_lines, base_tokens = _normalize_segments(tess_result, width, height)
        word_consensus = compute_word_consensus(
            base_tokens,
            {},
        )
        word_segments, line_segments = _build_segments(
            page_index,
            image_path,
            normalized_lines,
            word_consensus,
            other_outputs,
        )
        segments.extend(word_segments)
        segments.extend(line_segments)
        for line in line_segments:
            master_lines[(page_index, line.line_index)] = line.consensus_text

    segments_sorted = sorted(
        segments,
        key=lambda seg: (
            seg.page_index,
            seg.line_index,
            seg.word_index if seg.word_index is not None else -1,
            0 if seg.view == "line" else 1,
        ),
    )
    store.write_segments(segments_sorted)
    if master_lines:
        ordered_lines = [
            master_lines[key] for key in sorted(master_lines.keys(), key=lambda item: (item[0], item[1]))
        ]
        store.write_master("\n".join(ordered_lines))
    else:
        store.write_master("")
    if not store.read_state():
        store.write_state({"view": "line", "segment_id": segments_sorted[0].segment_id if segments_sorted else ""})


def _prepare_page_images(source_paths: list[Path], store: ProjectStore) -> list[Path]:
    """Copy or render source files into per-page PNG images under the project assets folder."""
    page_images: list[Path] = []
    page_counter = 0
    for source_path in source_paths:
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError("pdf2image is required to process PDF inputs.") from exc
            pages = convert_from_path(str(source_path))
            for page in pages:
                asset_name = Path(f"page_{page_counter:04d}.png")
                destination = store.assets_dir / asset_name
                page.convert("RGB").save(destination)
                page_images.append(asset_name)
                page_counter += 1
        else:
            with Image.open(source_path) as image:
                asset_name = Path(f"page_{page_counter:04d}.png")
                destination = store.assets_dir / asset_name
                image.convert("RGB").save(destination)
                page_images.append(asset_name)
                page_counter += 1
    return page_images


def _run_tesseract_with_logging(image_path: Path, languages: list[str]) -> TesseractResult:
    try:
        return run_tesseract(image_path, languages)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.error("Tesseract OCR failed for %s: %s", image_path, exc)
        raise


def _persist_model_outputs(store: ProjectStore, page_index: int, outputs: Mapping[str, str]) -> None:
    for model_name, text in outputs.items():
        model_dir = store.outputs_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / f"page_{page_index:04d}.txt"
        _ = out_path.write_text(text or "", encoding="utf-8")


def _normalize_segments(
    tess_result: TesseractResult,
    image_width: int,
    image_height: int,
) -> tuple[list[NormalizedLine], list[BaseToken]]:
    lines: list[NormalizedLine] = []
    base_tokens: list[BaseToken] = []
    if not tess_result.lines:
        default_bbox = {"x": 0, "y": 0, "w": image_width, "h": image_height}
        default_line = NormalizedLine(line_index=0, text=tess_result.text.strip(), bbox=default_bbox, words=[])
        lines.append(default_line)
        for idx, token in enumerate(default_line.text.split()):
            bbox = _allocate_bbox_for_token(default_bbox, idx, len(default_line.text.split()))
            word = NormalizedWord(text=token, bbox=bbox, line_index=0, word_index=idx)
            default_line.words.append(word)
            base_tokens.append(BaseToken(text=token, line_index=0, word_index=idx))
        return lines, base_tokens

    for line in tess_result.lines:
        bbox = {
            "x": max(line.left, 0),
            "y": max(line.top, 0),
            "w": max(line.width, 1),
            "h": max(line.height, 1),
        }
        normalized_line = NormalizedLine(line_index=line.line_index, text=line.text, bbox=bbox, words=[])
        if getattr(line, "words", None):
            for word in line.words:
                word_bbox = {
                    "x": max(word.left, 0),
                    "y": max(word.top, 0),
                    "w": max(word.width, 1),
                    "h": max(word.height, 1),
                }
                normalized_word = NormalizedWord(
                    text=word.text,
                    bbox=word_bbox,
                    line_index=line.line_index,
                    word_index=word.word_index,
                )
                normalized_line.words.append(normalized_word)
                base_tokens.append(BaseToken(text=word.text, line_index=line.line_index, word_index=word.word_index))
        if not normalized_line.words:
            tokens = [token for token in line.text.split() if token]
            total = len(tokens)
            for idx, token in enumerate(tokens):
                word_bbox = _allocate_bbox_for_token(bbox, idx, total)
                normalized_word = NormalizedWord(
                    text=token,
                    bbox=word_bbox,
                    line_index=line.line_index,
                    word_index=idx,
                )
                normalized_line.words.append(normalized_word)
                base_tokens.append(BaseToken(text=token, line_index=line.line_index, word_index=idx))
        lines.append(normalized_line)
    return lines, base_tokens


def _allocate_bbox_for_token(line_bbox: dict[str, int], index: int, total: int) -> dict[str, int]:
    if total <= 0:
        return dict(line_bbox)
    proportion = 1 / total
    x = line_bbox["x"] + int(line_bbox["w"] * proportion * index)
    width = int(line_bbox["w"] * proportion) or max(1, line_bbox["w"] // max(total, 1))
    return {"x": x, "y": line_bbox["y"], "w": width, "h": line_bbox["h"]}


def _build_segments(
    page_index: int,
    image_path: Path,
    lines: list[NormalizedLine],
    word_consensus: list[WordConsensus],
    _model_outputs: Mapping[str, str],
) -> tuple[list[Segment], list[Segment]]:
    word_segments: list[Segment] = []
    line_segments: list[Segment] = []

    consensus_by_line: dict[int, list[WordConsensus]] = {}
    for entry in word_consensus:
        consensus_by_line.setdefault(entry.line_index, []).append(entry)

    for line in lines:
        line_word_ids: list[str] = []
        consensus_entries = consensus_by_line.get(line.line_index, [])
        consensus_map = {
            (entry.line_index, entry.word_index): entry for entry in consensus_entries
        }
        for word in line.words:
            segment_id = _segment_id(page_index, line.line_index, word.word_index, view="word")
            consensus_entry = consensus_map.get((word.line_index, word.word_index))
            if consensus_entry:
                consensus_text = consensus_entry.display_text
                has_conflict = consensus_entry.has_conflict
                alternatives = dict(consensus_entry.alternatives)
            else:
                consensus_text = word.text
                has_conflict = False
                alternatives = {}
            word_segment = Segment(
                segment_id=segment_id,
                view="word",
                page_index=page_index,
                line_index=line.line_index,
                word_index=word.word_index,
                page_image=str(image_path),
                bbox=word.bbox,
                base_text=word.text,
                consensus_text=consensus_text,
                has_conflict=has_conflict,
                alternatives=alternatives,
                word_ids=[],
            )
            word_segments.append(word_segment)
            line_word_ids.append(segment_id)

        line_segment_id = _segment_id(page_index, line.line_index, None, view="line")
        if consensus_entries:
            consensus_entries_sorted = sorted(consensus_entries, key=lambda item: item.word_index)
            consensus_text = " ".join(entry.display_text for entry in consensus_entries_sorted).strip()
            has_conflict = any(entry.has_conflict for entry in consensus_entries_sorted)
            line_alternatives = _compose_line_alternatives(consensus_entries_sorted)
        else:
            base_tokens = [word.text for word in line.words]
            consensus_text = " ".join(base_tokens).strip()
            has_conflict = False
            line_alternatives = {}
        line_segment = Segment(
            segment_id=line_segment_id,
            view="line",
            page_index=page_index,
            line_index=line.line_index,
            word_index=None,
            page_image=str(image_path),
            bbox=line.bbox,
            base_text=line.text,
            consensus_text=consensus_text,
            has_conflict=has_conflict,
            alternatives=line_alternatives,
            word_ids=line_word_ids,
        )
        line_segments.append(line_segment)
    return word_segments, line_segments


def _segment_id(page_index: int, line_index: int, word_index: int | None, view: str) -> str:
    if view == "line":
        return f"p{page_index:03d}_l{line_index:04d}"
    return f"p{page_index:03d}_l{line_index:04d}_w{word_index or 0:04d}"


def _compose_line_alternatives(consensus_entries: list[WordConsensus]) -> dict[str, str]:
    alternatives: dict[str, list[str]] = {}
    for entry in consensus_entries:
        for model, text in entry.alternatives.items():
            token_list = alternatives.setdefault(model, [])
            token_list.append(text if text else entry.base)
    return {model: " ".join(tokens).strip() for model, tokens in alternatives.items()}
