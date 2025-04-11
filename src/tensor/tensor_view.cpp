#include "tensor/tensor_view.h"
#include "tensor/tensor_view.hpp"

namespace dlf {

template <typename T>
void TensorView<T>::set_value(const T& value) {
    if (is_const_) {
        throw std::runtime_error("Cannot modify a const TensorView");
    }
    data_[offset_] = value;
    value_ = value;
}

// Explicit instantiation for supported types
template class TensorView<double>;
template class TensorView<float>;
template class TensorView<int>;
template class TensorView<long>;
template class TensorView<bool>;

} // namespace dlf 