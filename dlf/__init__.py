"""
Python Deep Learning Framework
"""

from .types import (
    bool, int8, int16, int32, int64, int, long,
    float16, float32, float64, float, double)

from .tensor import Tensor
from .device import Device, DeviceType

__all__ = [
    # Scalar types
    "bool", "int8", "int16", "int32", "int64", "int", "long",
    "float16", "float32", "float64", "float", "double",

    # Device types
    "Device",
    "DeviceType",

    # Tensor
    "Tensor"
]
