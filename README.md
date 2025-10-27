# Lekha

Lekha is a CLI-driven workflow for manuscript transcription that runs OCR, highlights disagreements, and launches a lightweight web editor for human review.

## Features

- Run OCR over images or PDFs with Tesseract (additional engines can be added later).
- Compute word-level consensus and conflicts for targeted review.
- Preserve per-word metadata so you can plug in additional OCR engines for double-blind comparison in the future.
- Store project data in the appropriate XDG/AppData location for later resumption.
- Launch a local web viewer to navigate line-by-line or word-by-word, view cropped manuscript images, and correct transcription text.

## Quickstart

```bash
python -m pip install -e .
lekha /path/to/manuscript -l eng
```

If you invoke `lekha` without arguments, it will offer previously processed projects to resume.

> **Note**  
> Tesseract must be installed separately and available on your PATH. Lekha uses `pytesseract` when possible and falls back to the `tesseract` CLI.

The CLI keeps the `--model` flag so you can plug in a second OCR engine later; at the moment only `tesseract` is supported.

## Development

- `typer` powers the CLI.
- `Flask` serves the API and static assets for the web viewer.
- Tests are not included yet; verify manually by running the CLI and exercising the viewer.
