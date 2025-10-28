"""Flask application factory for the Lekha web viewer."""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
from pathlib import Path
from typing import cast

from flask import Flask, abort, jsonify, request, send_file, send_from_directory, session
from flask.typing import ResponseReturnValue
from PIL import Image

from .project import JSONValue, ProjectStore, Segment
from .config import get_data_root

logger = logging.getLogger(__name__)


def get_or_generate_secret_key() -> str:
    """
    Get secret key from environment or generate a new one.
    Warns if using the default development key.
    Returns a cryptographically secure secret key.
    """
    env_secret = os.environ.get("LEKHA_WEB_SECRET")
    if env_secret:
        if env_secret == "lekha-dev":
            logger.warning(
                "Using default secret key 'lekha-dev'. " +
                "Set LEKHA_WEB_SECRET environment variable to a secure random value in production."
            )
        return env_secret

    # Generate a secure random key
    secret_key = secrets.token_hex(32)
    logger.info("Generated new secret key for this session. Set LEKHA_WEB_SECRET to persist sessions across restarts.")
    return secret_key


def create_app(store: ProjectStore) -> Flask:
    static_dir = Path(__file__).parent / "web" / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    app.secret_key = get_or_generate_secret_key()
    default_project_id = store.project_id
    runtime_cache: dict[str, ProjectRuntime] = {default_project_id: ProjectRuntime(store)}

    def _payload_dict(raw: object, *, error_message: str) -> dict[str, object]:
        if not isinstance(raw, dict):
            abort(400, error_message)
        result: dict[str, object] = {}
        for key, value in cast(dict[object, object], raw).items():
            if isinstance(key, str):
                result[key] = value
        return result

    def ensure_runtime(project_id: str) -> ProjectRuntime:
        if not project_id:
            raise RuntimeError("Project identifier is required.")
        if project_id in runtime_cache:
            return runtime_cache[project_id]
        project_root = get_data_root() / project_id
        if not project_root.exists():
            raise RuntimeError(f"Project '{project_id}' not found.")
        tentative_store = ProjectStore(project_id)
        if not tentative_store.segments_path.exists():
            raise RuntimeError(f"Project '{project_id}' has no processed segments.")
        runtime_cache[project_id] = ProjectRuntime(tentative_store)
        return runtime_cache[project_id]

    def current_runtime() -> ProjectRuntime:
        session_value = session.get("project_id")
        project_id = session_value if isinstance(session_value, str) and session_value else default_project_id
        session["project_id"] = project_id
        try:
            runtime = ensure_runtime(project_id)
        except RuntimeError as exc:
            abort(400, str(exc))
        return runtime

    @app.get("/")
    def index() -> ResponseReturnValue:  # pyright: ignore[reportUnusedFunction]
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/state")
    def get_state() -> ResponseReturnValue:  # pyright: ignore[reportUnusedFunction]
        runtime = current_runtime()
        state = runtime.ensure_state()
        payload: dict[str, object] = {key: value for key, value in state.items()}
        payload["project_id"] = runtime.store.project_id
        return jsonify(payload)

    @app.get("/api/segment/<segment_id>")
    def get_segment(segment_id: str) -> ResponseReturnValue:  # pyright: ignore[reportUnusedFunction]
        runtime = current_runtime()
        seg = runtime.get_segment(segment_id)
        view_arg = request.args.get("view")
        view = view_arg if isinstance(view_arg, str) else None
        payload = runtime.segment_payload(seg.segment_id, view=view)
        return jsonify(payload)

    @app.get("/api/segment/<segment_id>/image")
    def segment_image(segment_id: str) -> ResponseReturnValue:  # pyright: ignore[reportUnusedFunction]
        runtime = current_runtime()
        seg = runtime.get_segment(segment_id)
        try:
            crop_bounds = runtime.get_crop_bounds(segment_id)
            image = runtime.load_segment_image(seg, crop_bounds)
        except FileNotFoundError as exc:
            abort(404, str(exc))
        except RuntimeError as exc:
            abort(500, str(exc))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        _ = buffer.seek(0)
        response = send_file(buffer, mimetype="image/png")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.post("/api/save")
    def save_segment() -> ResponseReturnValue:  # pyright: ignore[reportUnusedFunction]
        runtime = current_runtime()
        raw_json = cast(object, request.get_json(force=True))
        data = _payload_dict(raw_json, error_message="Invalid payload.")
        segment_id_val = data.get("segment_id")
        view_val = data.get("view")
        text_val = data.get("text", "")
        action_val = data.get("action", "save")
        if not isinstance(segment_id_val, str):
            abort(400, "segment_id must be a string.")
        if not isinstance(view_val, str) or view_val not in {"line", "word"}:
            abort(400, "segment_id and view are required.")
        if not isinstance(text_val, str):
            abort(400, "text must be a string.")
        if not isinstance(action_val, str):
            abort(400, "action must be a string.")
        runtime.save(segment_id_val, view_val, text_val)
        next_id = runtime.navigate(view_val, segment_id_val, action_val)
        runtime.persist_state(view_val, next_id)
        payload = runtime.segment_payload(next_id, view=view_val)
        payload["view"] = view_val
        return jsonify(payload)

    @app.post("/api/view")
    def change_view() -> ResponseReturnValue:  # pyright: ignore[reportUnusedFunction]
        runtime = current_runtime()
        raw_json = cast(object, request.get_json(force=True))
        data = _payload_dict(raw_json, error_message="Invalid payload.")
        target_view_raw = data.get("view")
        current_segment_raw = data.get("segment_id")
        if isinstance(current_segment_raw, str) and current_segment_raw:
            current_segment = current_segment_raw
        else:
            current_segment = runtime.ensure_state()["segment_id"]
        if not isinstance(target_view_raw, str) or target_view_raw not in {"line", "word"}:
            abort(400, "view must be 'line' or 'word'.")
        target_view = target_view_raw
        next_id = runtime.switch_view(current_segment, target_view)
        payload = runtime.segment_payload(next_id, view=target_view)
        payload["view"] = target_view
        return jsonify(payload)

    @app.get("/api/projects")
    def list_projects() -> ResponseReturnValue:  # pyright: ignore[reportUnusedFunction]
        projects: list[dict[str, str]] = []
        data_root = get_data_root()
        for manifest_path in data_root.glob("*/manifest.json"):
            try:
                raw_value = cast(JSONValue, json.loads(manifest_path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if not isinstance(raw_value, dict):
                continue
            project_id = raw_value.get("project_id")
            if not isinstance(project_id, str):
                continue
            label_value = raw_value.get("source")
            label = label_value if isinstance(label_value, str) else project_id
            projects.append({"project_id": project_id, "label": label})
        projects.sort(key=lambda item: item.get("label") or item.get("project_id") or "")
        return jsonify({"projects": projects})

    @app.post("/api/project")
    def change_project() -> ResponseReturnValue:  # pyright: ignore[reportUnusedFunction]
        raw_json = cast(object, request.get_json(force=True))
        data = _payload_dict(raw_json, error_message="Invalid payload.")
        project_id = data.get("project_id")
        if not project_id:
            abort(400, "project_id is required.")
        if not isinstance(project_id, str):
            abort(400, "project_id must be a string.")
        session_value = session.get("project_id")
        current_id = session_value if isinstance(session_value, str) and session_value else default_project_id
        if project_id == current_id:
            runtime = current_runtime()
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
        try:
            runtime = ensure_runtime(project_id)
        except RuntimeError as exc:
            abort(400, str(exc))
        session["project_id"] = project_id
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
    def export_master() -> ResponseReturnValue:  # pyright: ignore[reportUnusedFunction]
        runtime = current_runtime()
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
        self.store: ProjectStore = store
        self.segments: list[Segment] = store.load_segments()
        if not self.segments:
            raise RuntimeError("No segments available. Run OCR processing first.")
        self.segments_by_id: dict[str, Segment] = {segment.segment_id: segment for segment in self.segments}
        self.orders: dict[str, list[str]] = {
            "line": self._ordered_ids("line"),
            "word": self._ordered_ids("word"),
        }
        self.parents: dict[str, str] = {}
        for segment in self.segments:
            if segment.view == "line":
                for word_id in segment.word_ids:
                    self.parents[word_id] = segment.segment_id
        self.edits: dict[str, str] = self.store.read_edits()
        self.state: dict[str, str] = self.store.read_state()
        self.page_dimensions: dict[str, tuple[int, int]] = {}
        self.crop_cache: dict[str, dict[str, int]] = {}
        _ = self.ensure_state()

    def _ordered_ids(self, view: str) -> list[str]:
        filtered = [seg for seg in self.segments if seg.view == view]
        filtered.sort(
            key=lambda seg: (
                seg.page_index,
                seg.line_index,
                seg.word_index if seg.word_index is not None else -1,
            )
        )
        return [seg.segment_id for seg in filtered]

    def ensure_state(self) -> dict[str, str]:
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

    def segment_payload(self, segment_id: str, view: str | None = None) -> dict[str, object]:
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
            "image_url": f"/api/segment/{segment.segment_id}/image?project={self.store.project_id}",
            "navigation": self.navigation_status(segment.segment_id, active_view),
        }

    def load_segment_image(self, segment: Segment, crop: dict[str, int] | None = None) -> Image.Image:
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
            tokens: list[str] = []
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

    def _next_issue(self, view: str, start_index: int) -> str | None:
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

    def get_crop_bounds(self, segment_id: str) -> dict[str, int]:
        if segment_id not in self.crop_cache:
            segment = self.get_segment(segment_id)
            crop = self._crop_geometry(segment)
            self.crop_cache[segment_id] = crop
        return self.crop_cache[segment_id]

    def _persist(self) -> None:
        self.store.write_edits(self.edits)
        master_text = self._compose_master_text()
        self.store.write_master(master_text)

    def _compose_master_text(self) -> str:
        lines: list[str] = []
        for line_id in self.orders["line"]:
            lines.append(self.get_text(line_id))
        return "\n".join(lines)

    def navigation_status(self, segment_id: str, view: str) -> dict[str, bool]:
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

    def _get_page_dimensions(self, page_image: str) -> tuple[int, int]:
        """
        Lazily load page dimensions for an image.
        Dimensions are cached after first load.
        Raises FileNotFoundError or RuntimeError if image cannot be loaded.
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

    def _crop_geometry(
        self, segment: Segment, padding_x_ratio: float = 0.1, padding_y_ratio: float = 0.5
    ) -> dict[str, int]:
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
