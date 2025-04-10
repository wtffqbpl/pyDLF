"""
Tensor module for pyDLF
Provides Python bindings for the C++ Tensor class
"""

import pydlf._pydlf as _pydlf
import numpy as np

class Tensor:
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
            self._tensor = _pydlf.Tensor(shape, 0.0)
        else:
            if isinstance(data, (list, tuple)):
                data = np.array(data, dtype=dtype)
            if isinstance(data, np.ndarray):
                data = data.flatten().tolist()
                if not data:
                    self._tensor = _pydlf.Tensor(shape, 0.0)
                else:
                    self._tensor = _pydlf.Tensor(shape, data[0])
                    for i, value in enumerate(data):
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
            return TensorView(self._tensor.view(key))
        elif isinstance(key, tuple):
            view = self._tensor
            for k in key:
                view = view.view(k)
            return TensorView(view)
        raise TypeError("Invalid index type")
    
    def __setitem__(self, key, value):
        """Set a value in the tensor"""
        if isinstance(key, int):
            indices = []
            remaining = key
            for dim in reversed(self.shape):
                indices.insert(0, remaining % dim)
                remaining //= dim
            self._tensor.set_at(indices, float(value))
        elif isinstance(key, tuple):
            view = self._tensor
            for k in key[:-1]:
                view = view.view(k)
            view.set_at([key[-1]], float(value))
        else:
            raise TypeError("Invalid index type")
    
    def __repr__(self):
        """String representation of the tensor"""
        return f"Tensor(shape={self.shape})"
    
    def to_numpy(self):
        """Convert tensor to numpy array"""
        return np.array(self._tensor.data()).reshape(self.shape)

class TensorView:
    """Python wrapper for the C++ TensorView class"""
    
    def __init__(self, tensor_view):
        """
        Initialize a TensorView
        
        Args:
            tensor_view: C++ TensorView object
        """
        self._view = tensor_view
    
    def __getitem__(self, key):
        """Get a value from the view"""
        if isinstance(key, int):
            return self._view.at([key])
        elif isinstance(key, tuple):
            view = self._view
            for k in key[:-1]:
                view = view.view(k)
            return view.at([key[-1]])
        raise TypeError("Invalid index type")
    
    def __setitem__(self, key, value):
        """Set a value in the view"""
        if isinstance(key, int):
            self._view.set_at([key], float(value))
        elif isinstance(key, tuple):
            view = self._view
            for k in key[:-1]:
                view = view.view(k)
            view.set_at([key[-1]], float(value))
        else:
            raise TypeError("Invalid index type")
    
    def __repr__(self):
        """String representation of the view"""
        return "TensorView()" 