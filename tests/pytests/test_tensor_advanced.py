import numpy as np
import pytest
from dlf import Tensor, TensorDouble, TensorInt, TensorLong, TensorBool

def test_tensor_types():
    # Test different tensor types
    t_float = Tensor([2, 2])
    t_double = TensorDouble([2, 2])
    t_int = TensorInt([2, 2])
    t_long = TensorLong([2, 2])
    t_bool = TensorBool([2, 2])
    
    assert isinstance(t_float, Tensor)
    assert isinstance(t_double, TensorDouble)
    assert isinstance(t_int, TensorInt)
    assert isinstance(t_long, TensorLong)
    assert isinstance(t_bool, TensorBool)

def test_tensor_reshape():
    # Test tensor reshaping
    data = list(range(1, 9))
    t = Tensor([2, 4], data)
    
    # Reshape to different dimensions
    t.reshape([4, 2])
    assert t.shape() == [4, 2]
    assert t.size() == 8
    
    # Test invalid reshape
    with pytest.raises(ValueError):
        t.reshape([3, 3])  # Invalid size

def test_tensor_transform():
    # Test tensor transformation
    data = [1, 2, 3, 4]
    t = Tensor([2, 2], data)
    
    # Transform using a lambda function
    t.transform(lambda x: x * 2)
    assert t.at([0, 0]) == 2
    assert t.at([0, 1]) == 4
    assert t.at([1, 0]) == 6
    assert t.at([1, 1]) == 8

def test_tensor_permute():
    # Test tensor permutation
    data = [1, 2, 3, 4, 5, 6]
    t = Tensor([2, 3], data)
    
    # Permute dimensions
    t.permute([1, 0])
    assert t.shape() == [3, 2]
    assert t.at([0, 0]) == 1
    assert t.at([0, 1]) == 4
    assert t.at([1, 0]) == 2
    assert t.at([1, 1]) == 5
    assert t.at([2, 0]) == 3
    assert t.at([2, 1]) == 6

def test_tensor_device():
    # Test tensor device operations
    t = Tensor([2, 2])
    
    # Test CPU device
    assert t.device().type().name.lower() == "cpu"
    
    # Test CUDA device if available
    try:
        t.to("cuda")
        assert t.device().type().name.lower() == "cuda"
    except RuntimeError:
        pytest.skip("CUDA not available")

def test_tensor_strides():
    # Test tensor strides
    data = [1, 2, 3, 4, 5, 6]
    t = Tensor([2, 3], data)
    
    strides = t.strides()
    assert len(strides) == 2
    assert strides[0] == 3  # Stride for first dimension
    assert strides[1] == 1  # Stride for second dimension

def test_tensor_ndim():
    # Test tensor dimensionality
    t1 = Tensor([2, 3])
    assert t1.ndim() == 2
    
    t2 = Tensor([2, 3, 4])
    assert t2.ndim() == 3
    
    t3 = Tensor([5])
    assert t3.ndim() == 1

def test_tensor_empty():
    # Test empty tensor operations
    t = Tensor([0])
    assert t.empty()
    assert t.size() == 0
    
    t2 = Tensor([2, 0])
    assert t2.empty()
    assert t2.size() == 0

def test_tensor_view_advanced():
    # Test advanced tensor view operations
    data = list(range(1, 9))
    t = Tensor([2, 2, 2], data)
    
    # Test multiple views
    v1 = t.view(0)
    assert v1.remaining_dims() == [2, 2]
    
    # Test view modification
    v1.set_at([0, 0], 10)
    assert t.at([0, 0, 0]) == 10

def test_tensor_numpy_conversion():
    # Test numpy array conversion with different types
    # Float tensor
    t_float = Tensor([2], [1.5, 2.5])
    arr_float = t_float.to_numpy()
    assert arr_float.dtype == np.float32
    
    # Double tensor
    t_double = TensorDouble([2], [1.5, 2.5])
    arr_double = t_double.to_numpy()
    assert arr_double.dtype == np.float64
    
    # Int tensor
    t_int = TensorInt([2], [1, 2])
    arr_int = t_int.to_numpy()
    assert arr_int.dtype == np.int32
    
    # Long tensor
    t_long = TensorLong([2], [1, 2])
    arr_long = t_long.to_numpy()
    assert arr_long.dtype == np.int64
    
    # Bool tensor
    t_bool = TensorBool([2], [True, False])
    arr_bool = t_bool.to_numpy()
    assert arr_bool.dtype == np.bool_ 