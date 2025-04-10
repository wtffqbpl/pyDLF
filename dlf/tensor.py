"""
Python wrapper for the C++ Tensor class
"""

import numpy as np


class tensor:
    """Python wrapper for the C++ Tensor class"""

    def __init__(self, shape=None, data=None):
        """
        Initialize a Tensor

        Args:
            shape: Tuple of dimensions
            data: List of values
        """
        from ._pydlf import Tensor
        shape_list = list(shape) if shape else []
        
        # Create empty tensor with shape
        self._tensor = Tensor(shape_list)
        
        # Fill with data if provided
        if data:
            for idx, value in enumerate(data):
                # Convert flat index to multi-dimensional indices
                indices = []
                remaining = idx
                for dim in reversed(shape_list):
                    indices.insert(0, remaining % dim)
                    remaining //= dim
                self._tensor.set_at(indices, float(value))

    @property
    def shape(self):
        """Get the shape of the tensor"""
        return tuple(self._tensor.shape())

    def __getitem__(self, key):
        """Get a value from the tensor"""
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
                msg = (
                    f"Too many indices ({len(indices)}) for tensor with "
                    f"{len(self._tensor.shape())} dimensions"
                )
                raise ValueError(msg)
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
                remaining = len(self._tensor.shape()) - len(self._indices)
                msg = (
                    f"Expected {remaining} indices but got {len(key)}"
                )
                raise ValueError(msg)
        else:
            raise TypeError("Invalid index type")

    def __repr__(self):
        """String representation of the view"""
        dims = len(self._tensor.shape()) - len(self._indices)
        return f"tensor_view(indices={self._indices}, remaining_dims={dims})"

    def __float__(self):
        """Convert to float if possible"""
        if self._view.remaining_dims() == 0:
            return float(self._view.value())
        raise ValueError("Cannot convert multi-dimensional view to float") 