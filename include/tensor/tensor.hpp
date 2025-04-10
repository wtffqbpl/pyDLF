#pragma once

#include <algorithm>
#include <sstream>
#include <tuple>
#include "tensor.h"

namespace dlf
{

// TensorView implementation
template <typename T>
TensorView<T>::TensorView(Tensor<T>& tensor, size_t index)
    : tensor_(tensor), indices_(), is_const_(false)
{
    validate_index(index);
    indices_.push_back(index);
    remaining_shape_ = std::vector<size_t>(tensor.shape().begin() + 1, tensor.shape().end());
}

template <typename T>
TensorView<T>::TensorView(const Tensor<T>& tensor, size_t index)
    : tensor_(const_cast<Tensor<T>&>(tensor)), indices_(), is_const_(true)
{
    validate_index(index);
    indices_.push_back(index);
    remaining_shape_ = std::vector<size_t>(tensor.shape().begin() + 1, tensor.shape().end());
}

template <typename T>
TensorView<T>::TensorView(Tensor<T>& tensor, const std::vector<size_t>& indices)
    : tensor_(tensor), indices_(indices), is_const_(false)
{
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i >= tensor.shape().size() || indices[i] >= tensor.shape()[i]) {
            throw std::out_of_range("Index out of range");
        }
    }
    remaining_shape_ =
        std::vector<size_t>(tensor.shape().begin() + indices.size(), tensor.shape().end());
}

template <typename T>
TensorView<T> TensorView<T>::operator[](size_t index)
{
    validate_index(index);
    auto new_indices = indices_;
    new_indices.push_back(index);
    if (new_indices.size() == tensor_.shape().size()) {
        return TensorView<T>(tensor_, new_indices);
    }
    return TensorView<T>(tensor_, new_indices);
}

template <typename T>
const TensorView<T> TensorView<T>::operator[](size_t index) const
{
    validate_index(index);
    auto new_indices = indices_;
    new_indices.push_back(index);
    if (new_indices.size() == tensor_.shape().size()) {
        return TensorView<T>(tensor_, new_indices);
    }
    return TensorView<T>(tensor_, new_indices);
}

template <typename T>
T& TensorView<T>::at(const std::vector<size_t>& indices)
{
    if (is_const_) {
        throw std::runtime_error("Cannot modify const tensor view");
    }
    if (indices.size() != remaining_shape_.size()) {
        throw std::invalid_argument("Number of indices must match remaining dimensions");
    }
    auto new_indices = indices_;
    new_indices.insert(new_indices.end(), indices.begin(), indices.end());
    return tensor_.at(new_indices);
}

template <typename T>
const T& TensorView<T>::at(const std::vector<size_t>& indices) const
{
    if (indices.size() != remaining_shape_.size()) {
        throw std::invalid_argument("Number of indices must match remaining dimensions");
    }
    auto new_indices = indices_;
    new_indices.insert(new_indices.end(), indices.begin(), indices.end());
    return tensor_.at(new_indices);
}

template <typename T>
TensorView<T> TensorView<T>::view(size_t index)
{
    if (is_const_) {
        throw std::runtime_error("Cannot modify a const TensorView");
    }
    validate_index(index);
    auto new_indices = indices_;
    new_indices.push_back(index);
    return TensorView<T>(tensor_, new_indices);
}

template <typename T>
const TensorView<T> TensorView<T>::view(size_t index) const
{
    validate_index(index);
    auto new_indices = indices_;
    new_indices.push_back(index);
    return TensorView<T>(tensor_, new_indices);
}

template <typename T>
void TensorView<T>::validate_index(size_t index) const
{
    if (indices_.size() >= tensor_.shape().size()) {
        throw std::out_of_range("Index out of range: tensor view has no more dimensions");
    }
    if (index >= tensor_.shape()[indices_.size()]) {
        throw std::out_of_range("Index out of range: index exceeds dimension size");
    }
}

template <typename T>
size_t TensorView<T>::calculate_index() const
{
    std::vector<size_t> strides = tensor_.strides();
    size_t              index   = 0;
    for (size_t i = 0; i < indices_.size(); ++i) {
        index += indices_[i] * strides[i];
    }
    return index;
}

// Tensor implementation
template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, const std::vector<T>& data) : shape_(shape)
{
    validate_shape(shape);
    size_t total_size = calculate_size(shape);
    data_.reserve(total_size);
    if (data.empty()) {
        data_.resize(total_size, T());
    } else if (data.size() == total_size) {
        data_ = data;
    } else {
        throw std::invalid_argument("Data size does not match shape");
    }
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, const T& value) : shape_(shape)
{
    validate_shape(shape);
    size_t total_size = calculate_size(shape);
    data_.reserve(total_size);
    data_.resize(total_size, value);
}

template <typename T>
std::vector<size_t> Tensor<T>::strides() const
{
    std::vector<size_t> strides(shape_.size());
    size_t              stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape_[i];
    }
    return strides;
}

template <typename T>
T& Tensor<T>::operator[](size_t index)
{
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index];
}

template <typename T>
const T& Tensor<T>::operator[](size_t index) const
{
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index];
}

template <typename T>
TensorView<T> Tensor<T>::view(size_t index)
{
    return TensorView<T>(*this, index);
}

template <typename T>
const TensorView<T> Tensor<T>::view(size_t index) const
{
    return TensorView<T>(*this, index);
}

template <typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices)
{
    validate_indices(indices);
    return data_[calculate_index(indices)];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const
{
    validate_indices(indices);
    return data_[calculate_index(indices)];
}

template <typename T>
template <typename... Args>
T& Tensor<T>::operator()(Args... indices)
{
    std::vector<size_t> index_vec = {static_cast<size_t>(indices)...};
    validate_indices(index_vec);
    size_t flat_index = calculate_index(index_vec);
    return data_[flat_index];
}

template <typename T>
template <typename... Args>
const T& Tensor<T>::operator()(Args... indices) const
{
    std::vector<size_t> index_vec = {static_cast<size_t>(indices)...};
    validate_indices(index_vec);
    size_t flat_index = calculate_index(index_vec);
    return data_[flat_index];
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t>& new_shape)
{
    validate_shape(new_shape);
    if (calculate_size(new_shape) != data_.size()) {
        throw std::invalid_argument("New shape must have same total size");
    }
    shape_ = new_shape;
}

template <typename T>
void Tensor<T>::transform(std::function<T(const T&)> func)
{
    std::transform(data_.begin(), data_.end(), data_.begin(), func);
}

template <typename T>
std::vector<size_t> Tensor<T>::permute(const std::vector<size_t>& permutation)
{
    if (permutation.size() != shape_.size()) {
        throw std::invalid_argument("Permutation size must match tensor dimensions");
    }

    std::vector<bool> used(shape_.size(), false);
    for (size_t p : permutation) {
        if (p >= shape_.size() || used[p]) {
            throw std::invalid_argument("Invalid permutation");
        }
        used[p] = true;
    }

    std::vector<size_t> new_shape(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i) {
        new_shape[i] = shape_[permutation[i]];
    }

    shape_ = new_shape;
    return shape_;
}

template <typename T>
bool Tensor<T>::operator==(const Tensor& other) const
{
    return shape_ == other.shape_ && data_ == other.data_;
}

template <typename T>
bool Tensor<T>::operator!=(const Tensor& other) const
{
    return !(*this == other);
}

template <typename T>
std::string Tensor<T>::serialize() const
{
    std::stringstream ss;
    ss << shape_.size() << " ";
    for (size_t dim : shape_) {
        ss << dim << " ";
    }
    for (const T& value : data_) {
        ss << value << " ";
    }
    return ss.str();
}

template <typename T>
Tensor<T> Tensor<T>::deserialize(const std::string& str)
{
    std::stringstream ss(str);
    size_t            dims;
    ss >> dims;

    std::vector<size_t> shape(dims);
    for (size_t i = 0; i < dims; ++i) {
        ss >> shape[i];
    }

    std::vector<T> data;
    T              value;
    while (ss >> value) {
        data.push_back(value);
    }

    return Tensor(shape, data);
}

template <typename T>
size_t Tensor<T>::calculate_size(const std::vector<size_t>& shape) const
{
    if (shape.empty()) {
        return 0;
    }
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
void Tensor<T>::validate_shape(const std::vector<size_t>& shape) const
{
    // Allow empty shapes for scalar tensors
    if (shape.empty()) {
        return;
    }
    // Allow zero dimensions for empty tensors
    for (size_t dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("Shape dimensions cannot be negative");
        }
    }
}

template <typename T>
size_t Tensor<T>::calculate_index(const std::vector<size_t>& indices) const
{
    size_t index  = 0;
    size_t stride = 1;
    for (int i = indices.size() - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= shape_[i];
    }
    return index;
}

template <typename T>
void Tensor<T>::validate_indices(const std::vector<size_t>& indices) const
{
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
}

}  // namespace dlf