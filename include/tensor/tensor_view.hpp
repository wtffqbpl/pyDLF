#pragma once

#include "tensor/tensor_view.h"
#include <stdexcept>

namespace dlf {

template<typename T>
TensorView<T>::TensorView(Tensor<T>& tensor, size_t index)
    : tensor_(tensor), indices_({index}) {
    if (index >= tensor.shape()[0]) {
        throw std::out_of_range("Index out of bounds");
    }
    remaining_dims_ = std::vector<size_t>(tensor.shape().begin() + 1, tensor.shape().end());
}

template<typename T>
TensorView<T>::TensorView(const Tensor<T>& tensor, size_t index)
    : tensor_(const_cast<Tensor<T>&>(tensor)), indices_({index}) {
    if (index >= tensor.shape()[0]) {
        throw std::out_of_range("Index out of bounds");
    }
    remaining_dims_ = std::vector<size_t>(tensor.shape().begin() + 1, tensor.shape().end());
}

template<typename T>
TensorView<T>::TensorView(Tensor<T>& tensor, const std::vector<size_t>& indices)
    : tensor_(tensor), indices_(indices) {
    if (indices.size() > tensor.shape().size()) {
        throw std::invalid_argument("Too many indices");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= tensor.shape()[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    remaining_dims_ = std::vector<size_t>(tensor.shape().begin() + indices.size(), tensor.shape().end());
}

template<typename T>
TensorView<T>::TensorView(const Tensor<T>& tensor, const std::vector<size_t>& indices)
    : tensor_(const_cast<Tensor<T>&>(tensor)), indices_(indices) {
    if (indices.size() > tensor.shape().size()) {
        throw std::invalid_argument("Too many indices");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= tensor.shape()[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
    remaining_dims_ = std::vector<size_t>(tensor.shape().begin() + indices.size(), tensor.shape().end());
}

template<typename T>
T& TensorView<T>::at(const std::vector<size_t>& indices) {
    if (indices.size() != remaining_dims_.size()) {
        throw std::invalid_argument("Number of indices must match remaining dimensions");
    }
    std::vector<size_t> full_indices = indices_;
    full_indices.insert(full_indices.end(), indices.begin(), indices.end());
    return tensor_.at(full_indices);
}

template<typename T>
const T& TensorView<T>::at(const std::vector<size_t>& indices) const {
    if (indices.size() != remaining_dims_.size()) {
        throw std::invalid_argument("Number of indices must match remaining dimensions");
    }
    std::vector<size_t> full_indices = indices_;
    full_indices.insert(full_indices.end(), indices.begin(), indices.end());
    return tensor_.at(full_indices);
}

template<typename T>
void TensorView<T>::set_at(const std::vector<size_t>& indices, T value) {
    if (indices.size() != remaining_dims_.size()) {
        throw std::invalid_argument("Number of indices must match remaining dimensions");
    }
    std::vector<size_t> full_indices = indices_;
    full_indices.insert(full_indices.end(), indices.begin(), indices.end());
    tensor_.set_at(full_indices, value);
}

template<typename T>
TensorView<T> TensorView<T>::view(size_t index) const {
    if (remaining_dims_.empty()) {
        throw std::runtime_error("Cannot create view of scalar tensor");
    }
    if (index >= remaining_dims_[0]) {
        throw std::out_of_range("Index out of bounds");
    }
    std::vector<size_t> new_indices = indices_;
    new_indices.push_back(index);
    return TensorView<T>(tensor_, new_indices);
}

template<typename T>
T& TensorView<T>::value() {
    if (!remaining_dims_.empty()) {
        throw std::runtime_error("Cannot get value of non-scalar tensor view");
    }
    return tensor_.at(indices_);
}

template<typename T>
const T& TensorView<T>::value() const {
    if (!remaining_dims_.empty()) {
        throw std::runtime_error("Cannot get value of non-scalar tensor view");
    }
    return tensor_.at(indices_);
}

template<typename T>
const std::vector<size_t>& TensorView<T>::remaining_dims() const {
    return remaining_dims_;
}

template<typename T>
TensorView<T>& TensorView<T>::operator=(T value) {
    if (!remaining_dims_.empty()) {
        throw std::runtime_error("Cannot assign value to non-scalar tensor view");
    }
    tensor_.set_at(indices_, value);
    return *this;
}

template<typename T>
bool TensorView<T>::operator==(const TensorView& other) const {
    if (remaining_dims_ != other.remaining_dims_) {
        return false;
    }
    if (remaining_dims_.empty()) {
        return value() == other.value();
    }
    // Compare all elements in the view
    std::vector<size_t> indices(remaining_dims_.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = 0;
    }
    bool equal = true;
    while (equal) {
        equal = at(indices) == other.at(indices);
        // Increment indices
        size_t pos = indices.size() - 1;
        while (pos < indices.size() && ++indices[pos] >= remaining_dims_[pos]) {
            indices[pos] = 0;
            --pos;
        }
        if (pos >= indices.size()) {
            break;
        }
    }
    return equal;
}

template<typename T>
bool TensorView<T>::operator!=(const TensorView& other) const {
    return !(*this == other);
}

template<typename T>
bool TensorView<T>::operator==(T value) const {
    if (!remaining_dims_.empty()) {
        return false;
    }
    return this->value() == value;
}

template<typename T>
bool TensorView<T>::operator!=(T value) const {
    return !(*this == value);
}

} // namespace dlf 