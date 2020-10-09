"""Base classes and utilities that are useful for deploying ML models."""

from ml_base.ml_model import MLModel

__version_info__ = ("0", "1", "0")
__version__ = ".".join(__version_info__)

__all__ = ["MLModel"]
