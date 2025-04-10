#pragma once

#include "tensor/tensor.h"
#include "tensor/tensor_view.h"
#include <algorithm>
#include <sstream>
#include <tuple>
#include <numeric>

namespace dlf
{

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, const std::vector<T>& data)
    : shape_(shape),
      data_(data.empty() ? std::vector<T>(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())) : data)
{
    if (shape.empty())
    {
        throw std::invalid_argument("Shape cannot be empty");
    }
    size_t expected_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    if (!data.empty() && data.size() != expected_size)
    {
        throw std::invalid_argument("Data size does not match shape");
    }
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, const T& value)
    : shape_(shape),
      data_(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), value)
{
    if (shape.empty())
    {
        throw std::invalid_argument("Shape cannot be empty");
    }
}

template <typename T>
std::vector<size_t> Tensor<T>::strides() const
{
    std::vector<size_t> strides(shape_.size());
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;)
    {
        strides[i] = stride;
        stride *= shape_[i];
    }
    return strides;
}

template <typename T>
T& Tensor<T>::operator[](size_t index)
{
    if (index >= data_.size())
    {
        throw std::out_of_range("Index out of range");
    }
    return data_[index];
}

template <typename T>
const T& Tensor<T>::operator[](size_t index) const
{
    if (index >= data_.size())
    {
        throw std::out_of_range("Index out of range");
    }
    return data_[index];
}

template <typename T>
TensorView<T> Tensor<T>::view(size_t index)
{
    if (index >= shape_[0])
    {
        throw std::out_of_range("Index out of range");
    }
    return TensorView<T>(*this, index);
}

template <typename T>
const TensorView<T> Tensor<T>::view(size_t index) const
{
    if (index >= shape_[0])
    {
        throw std::out_of_range("Index out of range");
    }
    return TensorView<T>(*this, index);
}

template <typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices)
{
    if (indices.size() != shape_.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }

    size_t index = 0;
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;)
    {
        if (indices[i] >= shape_[i])
        {
            throw std::out_of_range("Index out of range");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }

    return data_[index];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const
{
    return const_cast<Tensor<T>*>(this)->at(indices);
}

template <typename T>
void Tensor<T>::set_at(const std::vector<size_t>& indices, T value)
{
    at(indices) = value;
}

template <typename T>
template <typename... Args>
T& Tensor<T>::operator()(Args... indices)
{
    std::vector<size_t> idx = {static_cast<size_t>(indices)...};
    return at(idx);
}

template <typename T>
template <typename... Args>
const T& Tensor<T>::operator()(Args... indices) const
{
    std::vector<size_t> idx = {static_cast<size_t>(indices)...};
    return at(idx);
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t>& new_shape)
{
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    if (new_size != data_.size())
    {
        throw std::invalid_argument("New shape must have the same total size");
    }
    shape_ = new_shape;
}

template <typename T>
void Tensor<T>::transform(std::function<T(const T&)> func)
{
    for (auto& value : data_)
    {
        value = func(value);
    }
}

template <typename T>
std::vector<size_t> Tensor<T>::permute(const std::vector<size_t>& permutation)
{
    if (permutation.size() != shape_.size())
    {
        throw std::invalid_argument("Permutation size must match tensor dimensions");
    }

    std::vector<bool> used(shape_.size(), false);
    for (size_t i = 0; i < permutation.size(); ++i)
    {
        if (permutation[i] >= shape_.size() || used[permutation[i]])
        {
            throw std::invalid_argument("Invalid permutation");
        }
        used[permutation[i]] = true;
    }

    std::vector<size_t> new_shape(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i)
    {
        new_shape[i] = shape_[permutation[i]];
    }

    std::vector<T> new_data(data_.size());
    std::vector<size_t> old_strides = strides();
    std::vector<size_t> new_strides(shape_.size());
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;)
    {
        new_strides[i] = stride;
        stride *= new_shape[i];
    }

    for (size_t i = 0; i < data_.size(); ++i)
    {
        size_t old_index = 0;
        size_t temp = i;
        for (size_t j = 0; j < shape_.size(); ++j)
        {
            size_t coord = temp / new_strides[j];
            temp %= new_strides[j];
            old_index += coord * old_strides[permutation[j]];
        }
        new_data[i] = data_[old_index];
    }

    data_ = std::move(new_data);
    shape_ = std::move(new_shape);
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
    
    // Write shape
    ss << shape_.size() << " ";  // Number of dimensions
    for (size_t dim : shape_) {
        ss << dim << " ";
    }
    
    // Write data
    for (const T& value : data_) {
        ss << value << " ";
    }
    
    return ss.str();
}

template <typename T>
Tensor<T> Tensor<T>::deserialize(const std::string& str)
{
    std::stringstream ss(str);
    std::vector<size_t> shape;
    std::vector<T> data;
    
    // Read shape
    size_t num_dims;
    ss >> num_dims;
    shape.resize(num_dims);
    for (size_t i = 0; i < num_dims; ++i) {
        ss >> shape[i];
    }
    
    // Read data
    T value;
    while (ss >> value) {
        data.push_back(value);
    }
    
    return Tensor<T>(shape, data);
}

template <typename T>
size_t Tensor<T>::calculate_size(const std::vector<size_t>& shape) const
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

template <typename T>
void Tensor<T>::validate_shape(const std::vector<size_t>& shape) const
{
    if (shape.empty())
    {
        throw std::invalid_argument("Shape cannot be empty");
    }
}

template <typename T>
size_t Tensor<T>::calculate_index(const std::vector<size_t>& indices) const
{
    if (indices.size() != shape_.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }

    size_t index = 0;
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;)
    {
        if (indices[i] >= shape_[i])
        {
            throw std::out_of_range("Index out of range");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }

    return index;
}

template <typename T>
void Tensor<T>::validate_indices(const std::vector<size_t>& indices) const
{
    if (indices.size() != shape_.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] >= shape_[i])
        {
            throw std::out_of_range("Index out of range");
        }
    }
}

} // namespace dlf