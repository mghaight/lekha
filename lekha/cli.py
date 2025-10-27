"""Command-line interface for Lekha."""

from __future__ import annotations

import json
import shutil
import sys
import webbrowser
from pathlib import Path
from typing import Annotated, cast

import typer

from .config import get_data_root
from .project import ProjectManifest, ProjectStore, project_id_for_path
from .processing import process_inputs
from .server import create_app, run_server

app = typer.Typer(add_completion=False, invoke_without_command=True, help="Lekha manuscript OCR and editor")

MODELS_DEFAULT: tuple[str, ...] = ("tesseract",)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in cast(list[object], value):
        result.append(item if isinstance(item, str) else str(item))
    return result


def _list_projects() -> list[ProjectManifest]:
    manifests: list[ProjectManifest] = []
    for manifest_path in get_data_root().glob("*/manifest.json"):
        try:
            raw_obj = cast(object, json.loads(manifest_path.read_text(encoding="utf-8")))
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


def normalize_delete_args(args: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--delete":
            value = "prompt"
            if index + 1 < len(args) and not args[index + 1].startswith("-"):
                value = args[index + 1]
                index += 1
            normalized.extend(["--delete", value])
        else:
            normalized.append(token)
        index += 1
    return normalized


def _delete_project_directory(project_id: str) -> None:
    project_root = get_data_root() / project_id
    if not project_root.exists():
        typer.echo(f"Project '{project_id}' not found.")
        return
    shutil.rmtree(project_root)
    typer.echo(f"Deleted project '{project_id}'.")


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
    manuscript: Annotated[Path | None, typer.Argument(exists=True, readable=True, resolve_path=True)] = None,
    language: Annotated[list[str] | None, typer.Option("-l", "--lang", help="Languages passed to the OCR engines")] = None,
    models: Annotated[list[str] | None, typer.Option("--model", help="OCR models to run (currently only tesseract)")] = None,
    delete: Annotated[str | None, typer.Option("--delete", help="Delete a project (use value 'all' to remove every project)", show_default=False)] = None,
    port: Annotated[int, typer.Option(help="Port for the local web viewer")] = 8765,
    no_browser: Annotated[bool, typer.Option(help="Do not automatically open the browser")] = False,
) -> None:
    """
    Process a manuscript with OCR engines and launch the Lekha web viewer.

    When invoked without a manuscript path, Lekha will list existing projects to resume.
    """
    if delete is not None:
        projects = _list_projects()
        if not projects:
            typer.echo("No stored projects found.")
            raise typer.Exit()
        if delete.lower() == "all":
            if typer.confirm(f"Delete all {len(projects)} project(s)?", default=False):
                for manifest in projects:
                    _delete_project_directory(manifest.project_id)
            raise typer.Exit()
        if delete == "prompt":
            project_id = _choose_existing()
            if not project_id:
                raise typer.Exit(code=1)
        else:
            project_id = delete
        if typer.confirm(f"Delete project '{project_id}'?", default=False):
            _delete_project_directory(project_id)
        raise typer.Exit()

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
    model_values = list(models) if models else list(MODELS_DEFAULT)
    languages = language_values or ["eng"]
    project_id = project_id_for_path(manuscript if manuscript.is_dir() else manuscript.parent)
    typer.echo(f"Processing {len(source_paths)} file(s) for project {project_id}...")
    store = ProjectStore(project_id)
    process_inputs(source_paths, languages=languages, models=model_values, store=store, source=str(manuscript))
    app_instance = create_app(store)
    if not no_browser:
        _ = webbrowser.open(f"http://127.0.0.1:{port}")
    run_server(app_instance, port=port)


def run() -> None:
    args = normalize_delete_args(sys.argv[1:])
    app(prog_name="lekha", args=args)
