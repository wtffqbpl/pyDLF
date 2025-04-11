#pragma once

#include "tensor_view.h"
#include <numeric>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace dlf {

template <typename T>
TensorView<T>::TensorView(const std::vector<T>& data, size_t offset,
           const std::vector<size_t>& dims,
           const std::vector<size_t>& strides)
    : data_(const_cast<std::vector<T>&>(data)), offset_(offset),
      dims_(dims), strides_(strides),
      value_(data[offset]), is_const_(true) {}

template <typename T>
TensorView<T>::TensorView(std::vector<T>& data, size_t offset, const std::vector<size_t>& dims, const std::vector<size_t>& strides)
    : data_(data), offset_(offset), dims_(dims), strides_(strides),
      value_(data[offset]), is_const_(false) {}

template <typename T>
TensorView<T>::TensorView(const TensorView& other)
    : data_(other.data_), offset_(other.offset_),
      dims_(other.dims_), strides_(other.strides_),
      value_(other.value_), is_const_(other.is_const_) {}

template <typename T>
TensorView<T>& TensorView<T>::operator=(const TensorView& other) {
    if (this != &other) {
        data_ = other.data_;
        offset_ = other.offset_;
        dims_ = other.dims_;
        strides_ = other.strides_;
        value_ = other.value_;
        is_const_ = other.is_const_;
    }
    return *this;
}

template <typename T>
TensorView<T>& TensorView<T>::operator=(T value) {
    if (is_const_) {
        throw std::runtime_error("Cannot modify a const TensorView");
    }
    data_[offset_] = value;
    value_ = value;
    return *this;
}

template <typename T>
bool TensorView<T>::operator==(const TensorView& other) const {
    return value_ == other.value_;
}

template <typename T>
bool TensorView<T>::operator!=(const TensorView& other) const {
    return value_ != other.value_;
}

template <typename T>
bool TensorView<T>::operator==(T value) const {
    return value_ == value;
}

template <typename T>
bool TensorView<T>::operator!=(T value) const {
    return value_ != value;
}

template <typename T>
TensorView<T>::operator T() const {
    return value_;
}

template <typename T>
T& TensorView<T>::value() {
    if (is_const_) {
        throw std::runtime_error("Cannot modify a const TensorView");
    }
    if (dims_.size() != 0) {
        throw std::runtime_error("Cannot call value() on a non-scalar TensorView");
    }
    return data_[offset_];
}

template <typename T>
const T& TensorView<T>::value() const {
    if (dims_.size() != 0) {
        throw std::runtime_error("Cannot call value() on a non-scalar TensorView");
    }
    return data_[offset_];
}

template <typename T>
T& TensorView<T>::at(const std::vector<size_t>& indices) {
    if (is_const_) {
        throw std::runtime_error("Cannot modify a const TensorView");
    }
    return data_[offset_ + calculate_flat_index(indices)];
}

template <typename T>
const T& TensorView<T>::at(const std::vector<size_t>& indices) const {
    return data_[offset_ + calculate_flat_index(indices)];
}

template <typename T>
const std::vector<size_t>& TensorView<T>::shape() const {
    return dims_;
}

template <typename T>
const std::vector<size_t>& TensorView<T>::strides() const {
    return strides_;
}

template <typename T>
size_t TensorView<T>::size() const {
    return std::accumulate(dims_.begin(), dims_.end(), size_t{1}, std::multiplies<size_t>());
}

template <typename T>
size_t TensorView<T>::ndim() const {
    return dims_.size();
}

template <typename T>
TensorView<T> TensorView<T>::view(size_t index) {
    if (index >= dims_[0]) {
        throw std::out_of_range("Index out of range");
    }
    std::vector<size_t> new_dims(dims_.begin() + 1, dims_.end());
    std::vector<size_t> new_strides(strides_.begin() + 1, strides_.end());
    return TensorView<T>(data_, offset_ + index * strides_[0], new_dims, new_strides);
}

template <typename T>
const TensorView<T> TensorView<T>::view(size_t index) const {
    if (index >= dims_[0]) {
        throw std::out_of_range("Index out of range");
    }
    std::vector<size_t> new_dims(dims_.begin() + 1, dims_.end());
    std::vector<size_t> new_strides(strides_.begin() + 1, strides_.end());
    return TensorView<T>(data_, offset_ + index * strides_[0], new_dims, new_strides);
}

template <typename T>
size_t TensorView<T>::calculate_flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != dims_.size()) {
        throw std::invalid_argument("Number of indices does not match number of dimensions");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= dims_[i]) {
            throw std::out_of_range("Index out of range");
        }
    }
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides_[i];
    }
    return index;
}

// Bool specialization implementation
template<>
class TensorView<bool> {
public:
    TensorView(const std::vector<bool>& data, size_t offset,
               const std::vector<size_t>& dims,
               const std::vector<size_t>& strides)
        : data_(const_cast<std::vector<bool>&>(data)), offset_(offset),
          dims_(dims), strides_(strides) {}

    TensorView(std::vector<bool>& data, size_t offset,
               const std::vector<size_t>& dims,
               const std::vector<size_t>& strides)
        : data_(data), offset_(offset),
          dims_(dims), strides_(strides) {}

    TensorView(const TensorView& other)
        : data_(other.data_), offset_(other.offset_),
          dims_(other.dims_), strides_(other.strides_) {}

    TensorView& operator=(const TensorView& other) {
        if (this != &other) {
            data_ = other.data_;
            offset_ = other.offset_;
            dims_ = other.dims_;
            strides_ = other.strides_;
        }
        return *this;
    }

    bool value() const {
        return data_[offset_];
    }

    void set_value(bool value) {
        data_[offset_] = value;
    }

    bool at(const std::vector<size_t>& indices) const {
        return data_[offset_ + calculate_index(indices)];
    }

    void set_at(const std::vector<size_t>& indices, bool value) {
        data_[offset_ + calculate_index(indices)] = value;
    }

    const std::vector<size_t>& shape() const {
        return dims_;
    }

    const std::vector<size_t>& strides() const {
        return strides_;
    }

    size_t size() const {
        return std::accumulate(dims_.begin(), dims_.end(), size_t{1}, std::multiplies<size_t>());
    }

    size_t ndim() const {
        return dims_.size();
    }

    TensorView<bool> view(size_t index) {
        if (index >= dims_[0]) {
            throw std::out_of_range("Index out of range");
        }
        std::vector<size_t> new_dims(dims_.begin() + 1, dims_.end());
        std::vector<size_t> new_strides(strides_.begin() + 1, strides_.end());
        return TensorView<bool>(data_, offset_ + index * strides_[0], new_dims, new_strides);
    }

    const TensorView<bool> view(size_t index) const {
        if (index >= dims_[0]) {
            throw std::out_of_range("Index out of range");
        }
        std::vector<size_t> new_dims(dims_.begin() + 1, dims_.end());
        std::vector<size_t> new_strides(strides_.begin() + 1, strides_.end());
        return TensorView<bool>(data_, offset_ + index * strides_[0], new_dims, new_strides);
    }

protected:
    size_t calculate_index(const std::vector<size_t>& indices) const {
        if (indices.size() != dims_.size()) {
            throw std::invalid_argument("Number of indices does not match number of dimensions");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= dims_[i]) {
                throw std::out_of_range("Index out of range");
            }
        }
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides_[i];
        }
        return index;
    }

    std::vector<bool>& data_;
    size_t offset_;
    std::vector<size_t> dims_;
    std::vector<size_t> strides_;
};

}  // namespace dlf