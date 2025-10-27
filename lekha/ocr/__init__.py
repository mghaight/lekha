"""OCR engine abstractions for Lekha."""

from .tesseract_engine import run_tesseract, TesseractResult

__all__ = ["run_tesseract", "TesseractResult"]
