import numpy as np
import pytest
from dlf import Tensor


def test_tensor_creation():
    # Test creating a tensor with shape
    t1 = Tensor([2, 3])
    assert t1.shape() == [2, 3]
    assert t1.size() == 6
    assert not t1.empty()

    # Test creating a tensor with data
    data = [1, 2, 3, 4, 5, 6]
    t2 = Tensor([2, 3], data)
    assert t2.shape() == [2, 3]
    assert t2.size() == 6
    assert not t2.empty()


def test_tensor_access():
    # Create a tensor with data
    data = [1, 2, 3, 4, 5, 6]
    t = Tensor([2, 3], data)

    # Test accessing elements
    assert t.at([0, 0]) == 1
    assert t.at([0, 1]) == 2
    assert t.at([0, 2]) == 3
    assert t.at([1, 0]) == 4
    assert t.at([1, 1]) == 5
    assert t.at([1, 2]) == 6


def test_tensor_modification():
    # Create a tensor with data
    data = [1, 2, 3, 4, 5, 6]
    t = Tensor([2, 3], data)

    # Test modifying elements
    t.set_at([0, 0], 10)
    assert t.at([0, 0]) == 10

    t.set_at([1, 2], 20)
    assert t.at([1, 2]) == 20


def test_tensor_view():
    # Create a tensor with data
    data = [1, 2, 3, 4, 5, 6]
    t = Tensor([2, 3], data)

    # Test creating a view
    v = t.view(0)
    assert v.remaining_dims() == [3]
    assert v.at([0]) == 1
    assert v.at([1]) == 2
    assert v.at([2]) == 3

    # Test modifying through view
    v.set_at([0], 10)
    assert t.at([0, 0]) == 10


def test_tensor_to_numpy():
    # Create a tensor with data
    data = [1, 2, 3, 4, 5, 6]
    t = Tensor([2, 3], data)

    # Convert to numpy array
    arr = t.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    assert np.array_equal(arr, np.array([[1, 2, 3], [4, 5, 6]])) 