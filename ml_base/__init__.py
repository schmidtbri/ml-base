"""Base classes and utilities that are useful for deploying ML models."""
from os.path import abspath, dirname, join
from ml_base.ml_model import MLModel
from ml_base.decorator import MLModelDecorator
from ml_base.utilities.model_manager import ModelManager

__all__ = ["MLModel", "MLModelDecorator", "ModelManager"]


try:
    print(join(abspath(dirname(__file__)), "version.txt"))
    with open(join(abspath(dirname(__file__)), "version.txt"), encoding="utf-8") as f:
        __version__ = f.read()
except Exception:
    __version__ = "N/A"
