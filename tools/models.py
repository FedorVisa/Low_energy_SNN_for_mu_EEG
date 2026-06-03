"""Compatibility wrapper for the project model namespace."""

from src import models as _models
from src.models import EEGNet, FBCNet, ShallowConvNet, deepconv

__all__ = _models.__all__


def __getattr__(name):
    return getattr(_models, name)
