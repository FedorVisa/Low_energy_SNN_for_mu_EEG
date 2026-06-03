"""Model namespace for EEG baseline and spiking neural network architectures."""

from importlib import import_module

from .ConvNet import ShallowConvNet, deepconv
from .EEGNet import EEGNet
from .FBCNet import FBCNet

_LAZY_EXPORTS = {
    "CUPY_SNN_PLIF",
    "CUPY_SNN_2ALIF",
    "CUPY_SNN_2PLIF",
    "CUPY_SNN_ALIF_READOUT",
    "CUPY_SNN_LIF_READOUT",
    "CUPY_SNN_LIF_READOUT_STREAMING",
    "CUPY_SNN_SPIKING_CONV_LIF_READOUT",
    "CUPY_SNN_SIGNED_LIF_MLP_READOUT",
    "CUPY_SNN_LIF_PLIF_LIF_READOUT",
    "CUPY_SNN_3PLIF_LN_ALIF_READOUT",
    "CUPY_SNN_3PLIF_PARALLEL",
    "CUPY_SNN_3PLIF_PARALLEL_ALIF_READOUT",
    "CUPY_SNN_PLIF_DUAL_READOUT",
    "NORSE_LATENCY_CONV_LIF_READOUT",
}

__all__ = [
    "ShallowConvNet",
    "deepconv",
    "EEGNet",
    "FBCNet",
    *_LAZY_EXPORTS,
]


def __getattr__(name):
    if name == "NORSE_LATENCY_CONV_LIF_READOUT":
        module = import_module("src.models.norse_models")
        return getattr(module, name)
    if name in _LAZY_EXPORTS:
        module = import_module("src.models.SNNs")
        return getattr(module, name)
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
