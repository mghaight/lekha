"""Integration with Kraken OCR via CLI calls."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class KrakenResult:
    text: str


def run_kraken(image_path: Path, languages: Sequence[str]) -> KrakenResult:
    """Execute Kraken OCR via its CLI. Requires the `kraken` command to be available."""
    executable = shutil.which("kraken")
    if executable is None:
        # Provide a gentle fallback so the pipeline can continue.
        return KrakenResult(text="")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.txt"
        cmd = [
            executable,
            "-i",
            str(image_path),
            str(output_file),
        ]
        if languages:
            cmd += ["-l", ",".join(languages)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime failure path
            raise RuntimeError(f"Kraken OCR failed: {exc.stderr.decode('utf-8', errors='ignore')}") from exc
        if output_file.exists():
            text = output_file.read_text(encoding="utf-8")
        else:
            text = ""
    return KrakenResult(text=text)
