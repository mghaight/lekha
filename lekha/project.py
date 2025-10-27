"""Project persistence and metadata management."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .config import get_data_root


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
        raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Invalid manifest format.")
        data = cast(dict[str, Any], raw)
        return ProjectManifest(**data)

    def write_segments(self, segments: list[Segment]) -> None:
        payload = [segment.__dict__ for segment in segments]
        with self.segments_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def load_segments(self) -> list[Segment]:
        if not self.segments_path.exists():
            return []
        raw = json.loads(self.segments_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Invalid segments data.")
        items = cast(list[dict[str, Any]], raw)
        return [Segment(**item) for item in items]

    def read_edits(self) -> dict[str, str]:
        if not self.edits_path.exists():
            return {}
        raw = json.loads(self.edits_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Invalid edits data.")
        return cast(dict[str, str], raw)

    def write_edits(self, edits: dict[str, str]) -> None:
        with self.edits_path.open("w", encoding="utf-8") as fh:
            json.dump(edits, fh, indent=2)

    def read_state(self) -> dict[str, str]:
        if not self.state_path.exists():
            return {}
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Invalid state data.")
        return cast(dict[str, str], raw)

    def write_state(self, state: dict[str, str]) -> None:
        with self.state_path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    def write_master(self, text: str) -> None:
        self.master_path.write_text(text, encoding="utf-8")
