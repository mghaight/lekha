"""Integration with Kraken OCR via CLI calls."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


logger = logging.getLogger(__name__)


@dataclass
class KrakenResult:
    text: str


def run_kraken(image_path: Path, languages: Sequence[str], *, model: str | None = None) -> KrakenResult:
    """Execute Kraken OCR via its CLI.

    Kraken must be installed and discoverable on ``PATH``. If the command is not available,
    an empty result is returned so the pipeline can continue while logging a warning.
    """

    executable = shutil.which("kraken")
    if executable is None:
        logger.warning("Kraken CLI not found on PATH; skipping Kraken OCR output.")
        return KrakenResult(text="")

    model_override = model or os.environ.get("LEKHA_KRAKEN_MODEL")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.txt"
        cmd = [
            executable,
            "ocr",
            "-i",
            str(image_path),
            "-o",
            str(output_file),
            "--format",
            "text",
        ]
        if model_override:
            cmd += ["--model", model_override]
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime failure path
            stderr = exc.stderr or ""
            logger.error("Kraken OCR failed for %s: %s", image_path, stderr.strip())
            raise RuntimeError(f"Kraken OCR failed: {stderr.strip()}") from exc

        text = ""
        if output_file.exists():
            text = output_file.read_text(encoding="utf-8").strip()
        if not text and completed.stdout:
            text = completed.stdout.strip()

    if not text:
        logger.warning("Kraken OCR produced no text for %s. Check the installed model.", image_path)
    return KrakenResult(text=text)
