#include "tensor/tensor.h"
#include "tensor/tensor.hpp"

namespace dlf {

// Template instantiations
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;

template class TensorView<float>;
template class TensorView<double>;
template class TensorView<int>;

} // namespace dlf 