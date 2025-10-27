"""OCR engine abstractions for Lekha."""

from .tesseract_engine import run_tesseract, TesseractResult
from .kraken_engine import run_kraken, KrakenResult

__all__ = ["run_tesseract", "TesseractResult", "run_kraken", "KrakenResult"]
