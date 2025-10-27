"""Flask application factory for the Lekha web viewer."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image

from .project import ProjectStore, Segment
from .config import get_data_root


def create_app(store: ProjectStore) -> Flask:
    static_dir = Path(__file__).parent / "web" / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    runtime = ProjectRuntime(store)

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/state")
    def get_state():
        state = runtime.ensure_state()
        payload = dict(state)
        payload["project_id"] = runtime.store.project_id
        return jsonify(payload)

    @app.get("/api/segment/<segment_id>")
    def get_segment(segment_id: str):
        seg = runtime.get_segment(segment_id)
        view = request.args.get("view")
        payload = runtime.segment_payload(seg.segment_id, view=view)
        return jsonify(payload)

    @app.get("/api/segment/<segment_id>/image")
    def segment_image(segment_id: str):
        seg = runtime.get_segment(segment_id)
        crop_info = runtime.get_crop_info(segment_id)
        image = runtime.load_segment_image(seg, crop_info["crop"])
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    @app.post("/api/save")
    def save_segment():
        data = request.get_json(force=True)
        segment_id = data.get("segment_id")
        view = data.get("view")
        text = data.get("text", "")
        action = data.get("action", "save")
        if not segment_id or view not in {"line", "word"}:
            abort(400, "segment_id and view are required.")
        runtime.save(segment_id, view, text)
        next_id = runtime.navigate(view, segment_id, action)
        runtime.persist_state(view, next_id)
        payload = runtime.segment_payload(next_id, view=view)
        payload["view"] = view
        return jsonify(payload)

    @app.post("/api/view")
    def change_view():
        data = request.get_json(force=True)
        target_view = data.get("view")
        current_segment = data.get("segment_id") or runtime.ensure_state()["segment_id"]
        if target_view not in {"line", "word"}:
            abort(400, "view must be 'line' or 'word'.")
        next_id = runtime.switch_view(current_segment, target_view)
        payload = runtime.segment_payload(next_id, view=target_view)
        payload["view"] = target_view
        return jsonify(payload)

    @app.get("/api/projects")
    def list_projects():
        projects: List[Dict[str, str]] = []
        data_root = get_data_root()
        for manifest_path in data_root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                projects.append(
                    {
                        "project_id": manifest.get("project_id"),
                        "label": manifest.get("source") or manifest.get("project_id"),
                    }
                )
            except Exception:
                continue
        projects.sort(key=lambda item: item.get("label") or item.get("project_id") or "")
        return jsonify({"projects": projects})

    @app.post("/api/project")
    def change_project():
        nonlocal runtime
        data = request.get_json(force=True)
        project_id = data.get("project_id")
        if not project_id:
            abort(400, "project_id is required.")
        if project_id == runtime.store.project_id:
            state = runtime.ensure_state()
            segment_payload = (
                runtime.segment_payload(state["segment_id"], view=state["view"]) if state.get("segment_id") else None
            )
            return jsonify(
                {
                    "project_id": runtime.store.project_id,
                    "view": state["view"],
                    "segment_id": state["segment_id"],
                    "segment": segment_payload,
                }
            )
        project_root = get_data_root() / project_id
        if not project_root.exists():
            abort(400, f"Project '{project_id}' not found.")
        new_store = ProjectStore(project_id)
        if not new_store.segments_path.exists():
            abort(400, f"Project '{project_id}' has no processed segments.")
        try:
            runtime = ProjectRuntime(new_store)
        except RuntimeError as exc:
            abort(400, str(exc))
        state = runtime.ensure_state()
        segment_payload = (
            runtime.segment_payload(state["segment_id"], view=state["view"]) if state.get("segment_id") else None
        )
        return jsonify(
            {
                "project_id": runtime.store.project_id,
                "view": state["view"],
                "segment_id": state["segment_id"],
                "segment": segment_payload,
            }
        )

    @app.get("/api/export/master")
    def export_master():
        master_path = runtime.store.master_path
        if not master_path.exists():
            abort(404, "Master transcription is not available yet.")
        download_name = f"{runtime.store.project_id}_master.txt"
        return send_file(master_path, mimetype="text/plain", as_attachment=True, download_name=download_name)

    return app


def run_server(app: Flask, port: int = 8765) -> None:
    app.run(host="127.0.0.1", port=port, debug=False)


class ProjectRuntime:
    """In-memory helper that mediates between Flask routes and on-disk storage."""

    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.segments: List[Segment] = store.load_segments()
        if not self.segments:
            raise RuntimeError("No segments available. Run OCR processing first.")
        self.segments_by_id: Dict[str, Segment] = {segment.segment_id: segment for segment in self.segments}
        self.orders: Dict[str, List[str]] = {
            "line": self._ordered_ids("line"),
            "word": self._ordered_ids("word"),
        }
        self.parents: Dict[str, str] = {}
        for segment in self.segments:
            if segment.view == "line":
                for word_id in segment.word_ids:
                    self.parents[word_id] = segment.segment_id
        self.edits: Dict[str, str] = self.store.read_edits()
        self.state = self.store.read_state()
        self.page_dimensions: Dict[str, tuple[int, int]] = {}
        self.crop_cache: Dict[str, Dict[str, object]] = {}
        self._populate_page_dimensions()
        self.ensure_state()

    def _ordered_ids(self, view: str) -> List[str]:
        filtered = [seg for seg in self.segments if seg.view == view]
        filtered.sort(
            key=lambda seg: (
                seg.page_index,
                seg.line_index,
                seg.word_index if seg.word_index is not None else -1,
            )
        )
        return [seg.segment_id for seg in filtered]

    def ensure_state(self) -> Dict[str, str]:
        default_view = "line" if self.orders["line"] else "word"
        default_segment = ""
        if self.orders[default_view]:
            default_segment = self.orders[default_view][0]
        if not self.state:
            self.state = {"view": default_view, "segment_id": default_segment}
            self.store.write_state(self.state)
        if self.state["view"] not in {"line", "word"}:
            self.state["view"] = default_view
        if self.state["segment_id"] not in self.segments_by_id and default_segment:
            self.state["segment_id"] = default_segment
            self.store.write_state(self.state)
        return self.state

    def get_segment(self, segment_id: str) -> Segment:
        if segment_id not in self.segments_by_id:
            abort(404, f"Unknown segment {segment_id}")
        return self.segments_by_id[segment_id]

    def segment_payload(self, segment_id: str, view: Optional[str] = None) -> Dict[str, object]:
        segment = self.get_segment(segment_id)
        text = self.get_text(segment_id)
        resolved = segment_id in self.edits
        has_conflict = segment.has_conflict and not resolved
        active_view = view if view in {"line", "word"} else segment.view
        return {
            "segment_id": segment.segment_id,
            "view": segment.view,
            "text": text,
            "has_conflict": has_conflict,
            "image_url": f"/api/segment/{segment.segment_id}/image",
            "navigation": self.navigation_status(segment.segment_id, active_view),
        }

    def load_segment_image(self, segment: Segment, crop: Optional[Dict[str, int]] = None) -> Image.Image:
        image_path = self.store.assets_dir / segment.page_image
        crop_bounds = crop or self.get_crop_info(segment.segment_id)["crop"]
        with Image.open(image_path) as image:
            left = crop_bounds["left"]
            top = crop_bounds["top"]
            right = crop_bounds["right"]
            bottom = crop_bounds["bottom"]
            return image.crop((left, top, right, bottom)).copy()

    def save(self, segment_id: str, view: str, text: str) -> None:
        if view == "line":
            self._save_line(segment_id, text)
        else:
            self._save_word(segment_id, text)
        self._persist()

    def _save_line(self, line_id: str, text: str) -> None:
        line_segment = self.get_segment(line_id)
        tokens = text.split()
        word_ids = line_segment.word_ids
        if word_ids:
            if not tokens:
                tokens = ["" for _ in word_ids]
            if len(tokens) < len(word_ids):
                tokens.extend([""] * (len(word_ids) - len(tokens)))
            elif len(tokens) > len(word_ids):
                leading = tokens[: len(word_ids) - 1]
                trailing = " ".join(tokens[len(word_ids) - 1 :])
                tokens = leading + [trailing]
            for word_id, token in zip(word_ids, tokens):
                self.edits[word_id] = token
        self.edits[line_id] = text

    def _save_word(self, word_id: str, text: str) -> None:
        word_segment = self.get_segment(word_id)
        self.edits[word_id] = text
        parent_line_id = self.parents.get(word_id)
        if parent_line_id:
            parent_segment = self.get_segment(parent_line_id)
            tokens = []
            for child_id in parent_segment.word_ids:
                if child_id == word_id:
                    tokens.append(text)
                else:
                    tokens.append(self.edits.get(child_id, self.get_segment(child_id).consensus_text))
            line_text = " ".join(tokens).strip()
            self.edits[parent_line_id] = line_text
        else:
            # ensure master text still consistent
            self._recalculate_line_from_word(word_segment.line_index, word_segment.page_index)

    def _recalculate_line_from_word(self, line_index: int, page_index: int) -> None:
        line_id = f"p{page_index:03d}_l{line_index:04d}"
        if line_id not in self.segments_by_id:
            return
        line_segment = self.segments_by_id[line_id]
        tokens = [
            self.edits.get(word_id, self.segments_by_id[word_id].consensus_text) for word_id in line_segment.word_ids
        ]
        self.edits[line_id] = " ".join(tokens).strip()

    def navigate(self, view: str, current_id: str, action: str) -> str:
        order = self.orders.get(view, [])
        if not order:
            return current_id
        try:
            index = order.index(current_id)
        except ValueError:
            index = 0
        if action == "prev":
            index = max(index - 1, 0)
        elif action == "next":
            index = min(index + 1, len(order) - 1)
        elif action == "next_issue":
            next_issue_id = self._next_issue(view, index)
            return next_issue_id or current_id
        return order[index]

    def _next_issue(self, view: str, start_index: int) -> Optional[str]:
        order = self.orders.get(view, [])
        length = len(order)
        if not length:
            return None
        begin = max(start_index + 1, 0)
        for idx in range(begin, length):
            seg_id = order[idx]
            segment = self.segments_by_id[seg_id]
            if segment.has_conflict and seg_id not in self.edits:
                return seg_id
        return None

    def switch_view(self, current_segment: str, target_view: str) -> str:
        segment = self.segments_by_id.get(current_segment)
        if target_view == "line":
            if segment and segment.view == "line":
                target = current_segment
            elif segment and segment.view == "word":
                target = self.parents.get(current_segment, "")
            else:
                target = ""
            if not target and self.orders["line"]:
                target = self.orders["line"][0]
        else:
            if segment and segment.view == "word":
                target = current_segment
            elif segment and segment.view == "line" and segment.word_ids:
                target = segment.word_ids[0]
            else:
                target = ""
            if not target and self.orders["word"]:
                target = self.orders["word"][0]
        if not target:
            target = current_segment
        self.persist_state(target_view, target)
        return target

    def persist_state(self, view: str, segment_id: str) -> None:
        self.state = {"view": view, "segment_id": segment_id}
        self.store.write_state(self.state)

    def get_text(self, segment_id: str) -> str:
        if segment_id in self.edits:
            return self.edits[segment_id]
        segment = self.get_segment(segment_id)
        if segment.view == "line" and segment.word_ids:
            tokens = [self.get_text(word_id) for word_id in segment.word_ids]
            text = " ".join(tokens).strip()
            return text if text else segment.consensus_text
        return segment.consensus_text

    def get_crop_info(self, segment_id: str) -> Dict[str, object]:
        if segment_id not in self.crop_cache:
            segment = self.get_segment(segment_id)
            crop = self._crop_geometry(segment)
            self.crop_cache[segment_id] = {"crop": crop}
        return self.crop_cache[segment_id]

    def _persist(self) -> None:
        self.store.write_edits(self.edits)
        master_text = self._compose_master_text()
        self.store.write_master(master_text)

    def _compose_master_text(self) -> str:
        lines = []
        for line_id in self.orders["line"]:
            lines.append(self.get_text(line_id))
        return "\n".join(lines)

    def navigation_status(self, segment_id: str, view: str) -> Dict[str, bool]:
        if view not in {"line", "word"}:
            return {"can_prev": False, "can_next": False, "has_next_issue": False}
        order = self.orders.get(view, [])
        if not order:
            return {"can_prev": False, "can_next": False, "has_next_issue": False}
        try:
            index = order.index(segment_id)
        except ValueError:
            return {"can_prev": False, "can_next": False, "has_next_issue": False}
        can_prev = index > 0
        can_next = index < len(order) - 1
        has_next_issue = self._next_issue(view, index) is not None
        return {"can_prev": can_prev, "can_next": can_next, "has_next_issue": has_next_issue}

    def _populate_page_dimensions(self) -> None:
        for segment in self.segments:
            if segment.page_image in self.page_dimensions:
                continue
            image_path = self.store.assets_dir / segment.page_image
            with Image.open(image_path) as image:
                self.page_dimensions[segment.page_image] = image.size

    def _crop_geometry(self, segment: Segment, padding_x_ratio: float = 0.1, padding_y_ratio: float = 0.5):
        width, height = self.page_dimensions[segment.page_image]
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
