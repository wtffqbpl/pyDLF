"""
Tensor module for dlf
Provides Python bindings for the C++ Tensor class
"""

import dlf._pydlf as _dlf
import numpy as np

class tensor:
    """Python wrapper for the C++ Tensor class"""
    
    def __init__(self, shape=None, data=None, dtype=float):
        """
        Initialize a Tensor
        
        Args:
            shape (tuple, optional): Shape of the tensor
            data (list or numpy.ndarray, optional): Initial data
            dtype (type, optional): Data type (float or int)
        """
        if shape is None:
            shape = []
        if isinstance(shape, (list, tuple)):
            shape = list(shape)
        
        if data is None:
            self._tensor = _dlf.Tensor(shape, 0.0)
        else:
            if isinstance(data, (list, tuple)):
                data = np.array(data, dtype=dtype)
            if isinstance(data, np.ndarray):
                data = data.flatten().tolist()
                if not data:
                    self._tensor = _dlf.Tensor(shape, 0.0)
                else:
                    self._tensor = _dlf.Tensor(shape, data[0])
                    for i, value in enumerate(data[1:], 1):
                        indices = []
                        remaining = i
                        for dim in reversed(shape):
                            indices.insert(0, remaining % dim)
                            remaining //= dim
                        self._tensor.set_at(indices, float(value))
    
    @property
    def shape(self):
        """Get the shape of the tensor"""
        return tuple(self._tensor.shape())
    
    def __getitem__(self, key):
        """Get a view of the tensor"""
        if isinstance(key, int):
            if len(self.shape) == 1:
                return float(self._tensor.at([key]))
            return tensor_view(self._tensor, [key])
        elif isinstance(key, tuple):
            if len(key) == len(self.shape):
                return float(self._tensor.at(list(key)))
            return tensor_view(self._tensor, list(key))
        raise TypeError("Invalid index type")
    
    def __setitem__(self, key, value):
        """Set a value in the tensor"""
        if isinstance(key, int):
            if len(self.shape) == 1:
                self._tensor.set_at([key], float(value))
            else:
                raise ValueError("Cannot set value on multi-dimensional tensor")
        elif isinstance(key, tuple):
            if len(key) == len(self.shape):
                self._tensor.set_at(list(key), float(value))
            else:
                raise ValueError("Number of indices must match dimensions")
        else:
            raise TypeError("Invalid index type")
    
    def __repr__(self):
        """String representation of the tensor"""
        return f"tensor(shape={self.shape})"
    
    def to_numpy(self):
        """Convert tensor to numpy array"""
        data = self._tensor.data()
        if not data:
            return np.array([]).reshape(self.shape)
        return np.asarray(data).reshape(self.shape)

class tensor_view:
    """Python wrapper for the C++ TensorView class"""
    
    def __init__(self, tensor, indices):
        """
        Initialize a TensorView
        
        Args:
            tensor: C++ Tensor object
            indices: List of indices
        """
        self._tensor = tensor
        self._indices = indices
    
    def __getitem__(self, key):
        """Get a value from the view"""
        if isinstance(key, int):
            indices = self._indices + [key]
            if len(indices) == len(self._tensor.shape()):
                return float(self._tensor.at(indices))
            return tensor_view(self._tensor, indices)
        elif isinstance(key, tuple):
            indices = self._indices + list(key)
            if len(indices) == len(self._tensor.shape()):
                return float(self._tensor.at(indices))
            elif len(indices) < len(self._tensor.shape()):
                return tensor_view(self._tensor, indices)
            else:
                raise ValueError(f"Too many indices ({len(indices)}) for tensor with {len(self._tensor.shape())} dimensions")
        raise TypeError("Invalid index type")
    
    def __setitem__(self, key, value):
        """Set a value in the view"""
        if isinstance(key, int):
            indices = self._indices + [key]
            if len(indices) == len(self._tensor.shape()):
                self._tensor.set_at(indices, float(value))
            else:
                raise ValueError("Cannot set value on multi-dimensional view")
        elif isinstance(key, tuple):
            indices = self._indices + list(key)
            if len(indices) == len(self._tensor.shape()):
                self._tensor.set_at(indices, float(value))
            else:
                raise ValueError(f"Expected {len(self._tensor.shape()) - len(self._indices)} indices but got {len(key)}")
        else:
            raise TypeError("Invalid index type")
    
    def __repr__(self):
        """String representation of the view"""
        return f"tensor_view(indices={self._indices}, remaining_dims={len(self._tensor.shape()) - len(self._indices)})"
    
    def __float__(self):
        """Convert to float if possible"""
        if self._view.remaining_dims() == 0:
            return float(self._view.value())
        raise ValueError("Cannot convert multi-dimensional view to float") 