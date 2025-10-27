"""Command-line interface for Lekha."""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import cast

import typer

from .config import get_data_root
from .project import ProjectManifest, ProjectStore, project_id_for_path
from .processing import process_inputs
from .server import create_app, run_server

app = typer.Typer(add_completion=False, invoke_without_command=True, help="Lekha manuscript OCR and editor")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        result.append(item if isinstance(item, str) else str(item))
    return result


def _list_projects() -> list[ProjectManifest]:
    manifests: list[ProjectManifest] = []
    for manifest_path in get_data_root().glob("*/manifest.json"):
        try:
            raw_obj: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw_obj, dict):
            continue
        raw = cast(dict[str, object], raw_obj)
        project_id = raw.get("project_id")
        if not isinstance(project_id, str):
            continue
        source_value = raw.get("source")
        languages = _string_list(raw.get("languages"))
        models = _string_list(raw.get("models"))
        files = _string_list(raw.get("files"))
        manifest = ProjectManifest(
            project_id=project_id,
            source=str(source_value) if isinstance(source_value, str) else project_id,
            languages=languages,
            models=models,
            files=files,
        )
        manifests.append(manifest)
    return sorted(manifests, key=lambda m: m.project_id)


def _choose_existing() -> str | None:
    projects = _list_projects()
    if not projects:
        typer.echo("No existing projects. Provide a manuscript path to get started.")
        return None
    typer.echo("Available projects:")
    for idx, project in enumerate(projects, start=1):
        typer.echo(f"{idx}. {project.project_id} ({project.source})")
    choice = cast(str, typer.prompt("Select project number", default="1"))
    try:
        index = int(choice) - 1
        if 0 <= index < len(projects):
            return projects[index].project_id
    except ValueError:
        pass
    typer.echo("Invalid selection.")
    return None


@app.callback(invoke_without_command=True)
def main(
    _ctx: typer.Context,
    manuscript: Path | None = typer.Argument(
        None, exists=True, readable=True, resolve_path=True
    ),
    language: list[str] = typer.Option(
        [], "-l", "--lang", help="Languages passed to the OCR engines"
    ),
    models: list[str] = typer.Option(
        ["tesseract"], "--model", help="OCR models to run (currently only tesseract)"
    ),
    port: int = typer.Option(8765, help="Port for the local web viewer"),
    no_browser: bool = typer.Option(False, help="Do not automatically open the browser"),
) -> None:
    """
    Process a manuscript with OCR engines and launch the Lekha web viewer.

    When invoked without a manuscript path, Lekha will list existing projects to resume.
    """
    if manuscript is None:
        project_id = _choose_existing()
        if not project_id:
            raise typer.Exit(code=1)
        store = ProjectStore(project_id)
        app_instance = create_app(store)
        if not no_browser:
            _ = webbrowser.open(f"http://127.0.0.1:{port}")
        run_server(app_instance, port=port)
        raise typer.Exit()

    source_paths: list[Path] = []
    if manuscript.is_dir():
        for candidate in sorted(manuscript.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}:
                source_paths.append(candidate)
    else:
        source_paths.append(manuscript)

    if not source_paths:
        typer.echo("No supported input files found.", err=True)
        raise typer.Exit(code=1)

    language_values = list(language) if language else []
    model_values = list(models) if models else []
    languages = language_values or ["eng"]
    project_id = project_id_for_path(manuscript if manuscript.is_dir() else manuscript.parent)
    typer.echo(f"Processing {len(source_paths)} file(s) for project {project_id}...")
    store = ProjectStore(project_id)
    process_inputs(source_paths, languages=languages, models=model_values, store=store, source=str(manuscript))
    app_instance = create_app(store)
    if not no_browser:
        _ = webbrowser.open(f"http://127.0.0.1:{port}")
    run_server(app_instance, port=port)
