"""Lekha package initialization."""

from importlib import resources

__all__ = ["data_path"]


def data_path(*parts: str) -> str:
    """Return the absolute path to a packaged resource."""
    return str(resources.files(__package__).joinpath(*parts))
