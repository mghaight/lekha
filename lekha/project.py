"""Project persistence and metadata management."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from .config import get_data_root

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | dict[str, "JSONValue"] | list["JSONValue"]
JSONDict = dict[str, JSONValue]
JSONList = list[JSONValue]


def slugify(name: str) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.lower())
    return "-".join(filter(None, normalized.split("-")))


def project_id_for_path(path: Path) -> str:
    canon = path.resolve()
    slug = slugify(canon.stem)
    digest = hashlib.sha1(str(canon).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}" if slug else digest


@dataclass
class Segment:
    segment_id: str
    view: str  # "line" or "word"
    page_index: int
    line_index: int
    word_index: int | None
    page_image: str
    bbox: dict[str, int]  # x, y, w, h
    base_text: str
    consensus_text: str
    has_conflict: bool
    alternatives: dict[str, str] = field(default_factory=dict)
    word_ids: list[str] = field(default_factory=list)


@dataclass
class ProjectManifest:
    project_id: str
    source: str
    languages: list[str]
    models: list[str]
    files: list[str]


class ProjectStore:
    """Coordinates persistence to the on-disk project directory."""

    project_id: str
    root: Path
    outputs_dir: Path
    assets_dir: Path
    meta_path: Path
    segments_path: Path
    edits_path: Path
    state_path: Path
    master_path: Path

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.root = get_data_root() / project_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = self.root / "outputs"
        self.outputs_dir.mkdir(exist_ok=True)
        self.assets_dir = self.root / "assets"
        self.assets_dir.mkdir(exist_ok=True)
        self.meta_path = self.root / "manifest.json"
        self.segments_path = self.root / "segments.json"
        self.edits_path = self.root / "edits.json"
        self.state_path = self.root / "state.json"
        self.master_path = self.root / "master.txt"

    def write_manifest(self, manifest: ProjectManifest) -> None:
        with self.meta_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest.__dict__, fh, indent=2)

    def load_manifest(self) -> ProjectManifest | None:
        if not self.meta_path.exists():
            return None
        raw_value = cast(JSONValue, json.loads(self.meta_path.read_text(encoding="utf-8")))
        if not isinstance(raw_value, dict):
            raise ValueError("Invalid manifest format.")
        project_id = _require_str(raw_value.get("project_id"), "project_id")
        source_value = raw_value.get("source")
        source = source_value if isinstance(source_value, str) else project_id
        languages = _coerce_str_list(raw_value.get("languages"))
        models = _coerce_str_list(raw_value.get("models"))
        files = _coerce_str_list(raw_value.get("files"))
        return ProjectManifest(
            project_id=project_id,
            source=source,
            languages=languages,
            models=models,
            files=files,
        )

    def write_segments(self, segments: list[Segment]) -> None:
        payload = [segment.__dict__ for segment in segments]
        with self.segments_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def load_segments(self) -> list[Segment]:
        if not self.segments_path.exists():
            return []
        raw_value = cast(JSONValue, json.loads(self.segments_path.read_text(encoding="utf-8")))
        if not isinstance(raw_value, list):
            raise ValueError("Invalid segments data.")
        segments: list[Segment] = []
        for item in raw_value:
            if not isinstance(item, dict):
                raise ValueError("Invalid segment entry encountered.")
            segments.append(_segment_from_dict(item))
        return segments

    def read_edits(self) -> dict[str, str]:
        if not self.edits_path.exists():
            return {}
        raw_value = cast(JSONValue, json.loads(self.edits_path.read_text(encoding="utf-8")))
        if not isinstance(raw_value, dict):
            raise ValueError("Invalid edits data.")
        edits: dict[str, str] = {}
        for key, value in raw_value.items():
            if isinstance(value, str):
                edits[key] = value
        return edits

    def write_edits(self, edits: dict[str, str]) -> None:
        with self.edits_path.open("w", encoding="utf-8") as fh:
            json.dump(edits, fh, indent=2)

    def read_state(self) -> dict[str, str]:
        if not self.state_path.exists():
            return {}
        raw_value = cast(JSONValue, json.loads(self.state_path.read_text(encoding="utf-8")))
        if not isinstance(raw_value, dict):
            raise ValueError("Invalid state data.")
        state: dict[str, str] = {}
        for key, value in raw_value.items():
            if isinstance(value, str):
                state[key] = value
        if "view" not in state:
            state["view"] = "line"
        if "segment_id" not in state:
            state["segment_id"] = ""
        return state

    def write_state(self, state: dict[str, str]) -> None:
        with self.state_path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    def write_master(self, text: str) -> None:
        _ = self.master_path.write_text(text, encoding="utf-8")


def _coerce_str_list(value: JSONValue | None) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif item is not None:
            result.append(str(item))
    return result


def _require_str(value: JSONValue | None, field: str) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"Manifest field '{field}' must be a string.")


def _int_from_json(value: JSONValue | None, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _bool_from_json(value: JSONValue | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
    if isinstance(value, int):
        return value != 0
    return False


def _segment_from_dict(data: JSONDict) -> Segment:
    segment_id = _require_str(data.get("segment_id"), "segment_id")
    view = _require_str(data.get("view"), "view")
    page_index = _int_from_json(data.get("page_index"))
    line_index = _int_from_json(data.get("line_index"))
    raw_word_index = data.get("word_index")
    word_index = None if raw_word_index is None else _int_from_json(raw_word_index)
    page_image = _require_str(data.get("page_image"), "page_image")
    bbox_value = data.get("bbox")
    bbox: dict[str, int] = {}
    if isinstance(bbox_value, dict):
        for key, value in bbox_value.items():
            bbox[key] = _int_from_json(value)
    base_text = _require_str(data.get("base_text"), "base_text")
    consensus_text = _require_str(data.get("consensus_text"), "consensus_text")
    has_conflict = _bool_from_json(data.get("has_conflict"))
    alternatives_value = data.get("alternatives")
    alternatives: dict[str, str] = {}
    if isinstance(alternatives_value, dict):
        for key, value in alternatives_value.items():
            alternatives[key] = value if isinstance(value, str) else str(value)
    word_ids_value = data.get("word_ids")
    word_ids = _coerce_str_list(word_ids_value)
    if view == "word":
        word_ids = []
    return Segment(
        segment_id=segment_id,
        view=view,
        page_index=page_index,
        line_index=line_index,
        word_index=word_index,
        page_image=page_image,
        bbox=bbox,
        base_text=base_text,
        consensus_text=consensus_text,
        has_conflict=has_conflict,
        alternatives=alternatives,
        word_ids=word_ids,
    )
