#pragma once

#include "tensor/tensor.h"

namespace dlf::ops {

/** Flatten a tensor into a 1D tensor.
 * @param tensor The input tensor to be flattened.
 * @return A new tensor that is a flattened version of the input tensor.
 */
template<typename T>
Tensor<T> flatten(const Tensor<T>& tensor) {
    // Create a new tensor with shape [total_size]
    std::vector<size_t> new_shape = {tensor.size()};
    return Tensor<T>(new_shape, tensor.data());
}

} // namespace dlf::ops