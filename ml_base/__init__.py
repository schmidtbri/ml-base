"""Base classes and utilities that are useful for deploying ML models."""

from ml_base.ml_model import MLModel
from ml_base.decorator import MLModelDecorator
from ml_base.utilities.model_manager import ModelManager

__all__ = ["MLModel", "MLModelDecorator", "ModelManager"]

__version__ = "<version_placeholder>"
