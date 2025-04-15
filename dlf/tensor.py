#! coding: utf-8

from typing import List, Union, Optional, Any, Callable, Dict
import numpy as np
from .device import Device, DeviceType
from .types import DType as dtypes
from ._pydlf import Tensor as _Tensor
from ._pydlf import TensorDouble, TensorInt, TensorLong, TensorBool


class Tensor:
    """
    A tensor is a multi-dimensional array that is used to store and manipulate data.
    This class is a wrapper around the _Tensor class in the _pydlf module.
    """
    def __init__(
        self,
        shape: Union[List[int], tuple],
        data: Optional[Union[List[Any], np.ndarray, _Tensor]] = None,
        device: Optional[Device] = None,
        value: Optional[Any] = None,
        dtype: Optional[dtypes] = None,
    ):
        """
        Initialize a new tensor.

        Args:
            shape: The shape of the tensor.
            data: The data to initialize the tensor with.
            device: The device to store the tensor on.
            value: The value to initialize the tensor with.
            dtype: The data type of the tensor.
        """
        if device is None:
            device = Device.cpu()
        
        if data is not None:
            if isinstance(data, np.ndarray):
                data = data.tolist()
            elif isinstance(data, _Tensor):
                data = data.tolist()
            elif isinstance(data, list):
                raise ValueError(f"Would implement this later: {type(data)}")
            else:
                raise ValueError(f"Invalid data type: {type(data)}")

        if dtype is None:
            dtype = dtypes.float32

        if value is not None:
            data = np.full(shape, value, dtype=dtype)
        elif data is not None:
            data = np.array(data, dtype=dtype)
        else:
            data = np.empty(shape, dtype=dtype)

    def __str__(self):
        return f"Tensor(shape={self.shape}, data={self.data}, device={self.device})"
    
    @staticmethod
    def tensor_factory(shape, data=None, dtype=None, device=None):
        if dtype is None:
            dtype = dtypes.float32
        if device is None:
            device = Device.cpu()
        return Tensor(shape, data, dtype, device)
