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
from .runtime import ImageService, SegmentNavigator, SegmentEditor

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
    """Facade coordinating specialized services for project operations."""

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

        # Initialize specialized services
        self.image_service: ImageService = ImageService(store, self.segments_by_id)
        self.navigator: SegmentNavigator = SegmentNavigator(self.orders, self.segments_by_id, self.parents, self.edits, self.state, store)
        self.editor: SegmentEditor = SegmentEditor(self.orders, self.segments_by_id, self.parents, self.edits, store)

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
        text = self.editor.get_text(segment_id)
        resolved = segment_id in self.edits
        has_conflict = segment.has_conflict and not resolved
        active_view = view if view in {"line", "word"} else segment.view
        return {
            "segment_id": segment.segment_id,
            "view": segment.view,
            "text": text,
            "has_conflict": has_conflict,
            "image_url": f"/api/segment/{segment.segment_id}/image?project={self.store.project_id}",
            "navigation": self.navigator.navigation_status(segment.segment_id, active_view),
        }

    # Delegation methods to services

    def load_segment_image(self, segment: Segment, crop: dict[str, int] | None = None) -> Image.Image:
        """Delegate to image service."""
        return self.image_service.load_segment_image(segment, crop)

    def get_crop_bounds(self, segment_id: str) -> dict[str, int]:
        """Delegate to image service."""
        return self.image_service.get_crop_bounds(segment_id)

    def _get_page_dimensions(self, page_image: str) -> tuple[int, int]:
        """Delegate to image service (for backwards compatibility in tests)."""
        return self.image_service._get_page_dimensions(page_image)  # pyright: ignore[reportPrivateUsage]

    def save(self, segment_id: str, view: str, text: str) -> None:
        """Delegate to editor service."""
        self.editor.save(segment_id, view, text)

    def get_text(self, segment_id: str) -> str:
        """Delegate to editor service."""
        return self.editor.get_text(segment_id)

    def navigate(self, view: str, current_id: str, action: str) -> str:
        """Delegate to navigator service."""
        return self.navigator.navigate(view, current_id, action)

    def switch_view(self, current_segment: str, target_view: str) -> str:
        """Delegate to navigator service."""
        return self.navigator.switch_view(current_segment, target_view)

    def persist_state(self, view: str, segment_id: str) -> None:
        """Delegate to navigator service."""
        self.navigator.persist_state(view, segment_id)

    def navigation_status(self, segment_id: str, view: str) -> dict[str, bool]:
        """Delegate to navigator service."""
        return self.navigator.navigation_status(segment_id, view)
