import pytest
import numpy as np
import dlf

def test_tensor_creation():
    # Test empty tensor
    t1 = dlf.tensor()
    assert t1.shape == ()
    
    # Test tensor with shape
    t2 = dlf.tensor((2, 3))
    assert t2.shape == (2, 3)
    
    # Test tensor with data
    data = [1, 2, 3, 4, 5, 6]
    t3 = dlf.tensor((2, 3), data)
    assert t3.shape == (2, 3)
    assert t3[0, 0] == 1
    assert t3[1, 2] == 6

def test_tensor_access():
    data = [1, 2, 3, 4, 5, 6]
    t = dlf.tensor((2, 3), data)
    
    # Test single index access
    assert t[0][0] == 1
    assert t[1][2] == 6
    
    # Test tuple index access
    assert t[0, 0] == 1
    assert t[1, 2] == 6
    
    # Test out of bounds access
    with pytest.raises(IndexError):
        t[2, 0]
    with pytest.raises(IndexError):
        t[0, 3]

def test_tensor_modification():
    data = [1, 2, 3, 4, 5, 6]
    t = dlf.tensor((2, 3), data)
    
    # Test single index modification
    t[0][0] = 10
    assert t[0, 0] == 10
    
    # Test tuple index modification
    t[1, 2] = 20
    assert t[1, 2] == 20

def test_tensor_view():
    data = [1, 2, 3, 4, 5, 6]
    t = dlf.tensor((2, 3), data)
    
    # Test view creation
    view = t[0]
    assert view[0] == 1
    assert view[1] == 2
    assert view[2] == 3
    
    # Test view modification
    view[0] = 10
    assert t[0, 0] == 10

def test_tensor_to_numpy():
    data = [1, 2, 3, 4, 5, 6]
    t = dlf.tensor((2, 3), data)
    
    # Test conversion to numpy array
    arr = t.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    assert np.array_equal(arr, np.array([[1, 2, 3], [4, 5, 6]]))
    
    # Test modification through tensor interface
    t[0, 0] = 10
    arr = t.to_numpy()
    assert arr[0, 0] == 10 