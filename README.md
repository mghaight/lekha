# Lekha

Lekha is a CLI-driven workflow for manuscript transcription that combines multiple OCR engines, highlights disagreements, and launches a lightweight web editor for human review.

## Features

- Run multiple OCR engines (Tesseract and Kraken by default) against images or PDFs.
- Compute word-level consensus and conflicts for targeted review.
- Store project data in the appropriate XDG/AppData location for later resumption.
- Launch a local web viewer to navigate line-by-line or word-by-word, view cropped manuscript images, and correct transcription text.

## Quickstart

```bash
python -m pip install -e .
lekha /path/to/manuscript -l eng
```

If you invoke `lekha` without arguments, it will offer previously processed projects to resume.

> **Note**  
> OCR engines such as Tesseract and Kraken must be installed separately and available on your PATH. Lekha uses `pytesseract` when possible and falls back to the `tesseract` CLI. For Kraken, set the `LEKHA_KRAKEN_MODEL` environment variable to the path of the recognition model (`.mlmodel`) you want to use if the default model is not available in your Kraken installation.

## Development

- `typer` powers the CLI.
- `Flask` serves the API and static assets for the web viewer.
- Tests are not included yet; verify manually by running the CLI and exercising the viewer.
