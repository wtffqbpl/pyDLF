#pragma once

#include "tensor.h"
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <sstream>

namespace dlf {

// Bool specialization implementation
inline Tensor<bool>::Tensor(const std::vector<size_t>& dims)
    : dims_(dims), data_(std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>())), device_(Device::cpu()) {
    calculate_strides();
}

inline Tensor<bool>::Tensor(const std::vector<size_t>& dims, const std::vector<bool>& data)
    : dims_(dims), data_(data), device_(Device::cpu()) {
    if (data.size() != std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>())) {
        throw std::invalid_argument("Data size does not match dimensions");
    }
    calculate_strides();
}

inline Tensor<bool>::Tensor(const std::vector<size_t>& dims, const Device& device)
    : dims_(dims), data_(std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>())), device_(device) {
    calculate_strides();
}

inline Tensor<bool>::Tensor(const std::vector<size_t>& dims, bool value)
    : dims_(dims), data_(std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>()), value), device_(Device::cpu()) {
    calculate_strides();
}

inline bool Tensor<bool>::operator[](size_t index) const {
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index];
}

inline void Tensor<bool>::set(size_t index, bool value) {
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    data_[index] = value;
}

inline bool Tensor<bool>::at(const std::vector<size_t>& indices) const {
    return data_[calculate_flat_index(indices)];
}

inline void Tensor<bool>::set_at(const std::vector<size_t>& indices, bool value) {
    data_[calculate_flat_index(indices)] = value;
}

inline const std::vector<size_t>& Tensor<bool>::shape() const { return dims_; }
inline const std::vector<size_t>& Tensor<bool>::strides() const { return strides_; }
inline const std::vector<bool>& Tensor<bool>::data() const { return data_; }
inline size_t Tensor<bool>::size() const { return data_.size(); }
inline size_t Tensor<bool>::ndim() const { return dims_.size(); }
inline const Device& Tensor<bool>::device() const { return device_; }

inline void Tensor<bool>::to(Device device) {
    if (device_ == device) return;
    device_ = device;
    // TODO: Implement device transfer logic
}

inline std::string Tensor<bool>::serialize() const {
    // TODO: Implement serialization
    return "";
}

inline void Tensor<bool>::deserialize(const std::vector<uint8_t>& data_bytes) {
    std::string data_str(data_bytes.begin(), data_bytes.end());
    std::istringstream iss(data_str);
    size_t ndims;
    iss >> ndims;
    
    dims_.resize(ndims);
    for (size_t i = 0; i < ndims; ++i) {
        iss >> dims_[i];
    }
    
    size_t total_size = std::accumulate(dims_.begin(), dims_.end(), size_t{1}, std::multiplies<size_t>());
    data_.resize(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        bool value;
        iss >> value;
        data_[i] = value;
    }
    
    calculate_strides();
}

inline void Tensor<bool>::calculate_strides() {
    strides_.resize(dims_.size());
    if (dims_.empty()) return;
    
    strides_[dims_.size() - 1] = 1;
    for (int i = dims_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
}

inline size_t Tensor<bool>::calculate_flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != dims_.size()) {
        throw std::invalid_argument("Number of indices does not match tensor dimensions");
    }
    
    size_t flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= dims_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        flat_index += indices[i] * strides_[i];
    }
    return flat_index;
}

inline bool Tensor<bool>::empty() const {
    return data_.empty();
}

inline void Tensor<bool>::reshape(const std::vector<size_t>& new_dims) {
    size_t new_size = std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<size_t>());
    if (new_size != data_.size()) {
        throw std::invalid_argument("New shape must have the same total size");
    }
    dims_ = new_dims;
    calculate_strides();
}

inline void Tensor<bool>::transform(const std::function<bool(bool)>& func) {
    std::transform(data_.begin(), data_.end(), data_.begin(), func);
}

inline void Tensor<bool>::permute(const std::vector<size_t>& axes) {
    if (axes.size() != dims_.size()) {
        throw std::invalid_argument("Number of axes must match tensor dimensions");
    }
    
    // Calculate new dimensions
    std::vector<size_t> new_dims(dims_.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] >= dims_.size()) {
            throw std::out_of_range("Axis index out of range");
        }
        new_dims[i] = dims_[axes[i]];
    }
    
    // Create a temporary copy of the data
    std::vector<bool> new_data(data_.size());
    
    // Calculate new strides
    std::vector<size_t> new_strides(new_dims.size());
    if (!new_dims.empty()) {
        new_strides[new_dims.size() - 1] = 1;
        for (int i = new_dims.size() - 2; i >= 0; --i) {
            new_strides[i] = new_strides[i + 1] * new_dims[i + 1];
        }
    }
    
    // Permute the data
    std::vector<size_t> indices(dims_.size(), 0);
    for (size_t i = 0; i < data_.size(); ++i) {
        // Calculate the new index using the original strides
        size_t old_index = 0;
        for (size_t j = 0; j < dims_.size(); ++j) {
            old_index += indices[j] * strides_[j];
        }
        
        // Calculate the new index using the permuted dimensions
        size_t new_index = 0;
        for (size_t j = 0; j < axes.size(); ++j) {
            new_index += indices[axes[j]] * new_strides[j];
        }
        
        // Copy the data
        new_data[new_index] = data_[old_index];
        
        // Increment indices
        for (int j = dims_.size() - 1; j >= 0; --j) {
            indices[j]++;
            if (indices[j] < dims_[j]) {
                break;
            }
            indices[j] = 0;
        }
    }
    
    // Update the tensor state
    dims_ = std::move(new_dims);
    strides_ = std::move(new_strides);
    data_ = std::move(new_data);
}

inline bool Tensor<bool>::operator==(const Tensor<bool>& other) const {
    return dims_ == other.dims_ && data_ == other.data_;
}

inline bool Tensor<bool>::operator!=(const Tensor<bool>& other) const {
    return !(*this == other);
}

inline TensorView<bool> Tensor<bool>::view(size_t index) {
    if (index >= dims_[0]) {
        throw std::out_of_range("Index out of bounds");
    }
    std::vector<size_t> remaining_dims(dims_.begin() + 1, dims_.end());
    std::vector<size_t> remaining_strides(strides_.begin() + 1, strides_.end());
    return TensorView<bool>(data_, index * strides_[0], remaining_dims, remaining_strides);
}

inline const TensorView<bool> Tensor<bool>::view(size_t index) const {
    if (index >= dims_[0]) {
        throw std::out_of_range("Index out of bounds");
    }
    std::vector<size_t> remaining_dims(dims_.begin() + 1, dims_.end());
    std::vector<size_t> remaining_strides(strides_.begin() + 1, strides_.end());
    return TensorView<bool>(data_, index * strides_[0], remaining_dims, remaining_strides);
}

// General template implementation
template <typename T>
inline Tensor<T>::Tensor(const std::vector<size_t>& dims)
    : dims_(dims), data_(std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>())), device_(Device::cpu()) {
    calculate_strides();
}

template <typename T>
inline Tensor<T>::Tensor(const std::vector<size_t>& dims, const std::vector<T>& data)
    : dims_(dims), data_(data), device_(Device::cpu()) {
    if (data.size() != std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>())) {
        throw std::invalid_argument("Data size does not match dimensions");
    }
    calculate_strides();
}

template <typename T>
inline Tensor<T>::Tensor(const std::vector<size_t>& dims, const Device& device)
    : dims_(dims), data_(std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>())), device_(device) {
    calculate_strides();
}

template <typename T>
inline Tensor<T>::Tensor(const std::vector<size_t>& dims, T value)
    : dims_(dims), data_(std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>()), value), device_(Device::cpu()) {
    calculate_strides();
}

template <typename T>
inline T& Tensor<T>::operator[](size_t index) {
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index];
}

template <typename T>
inline const T& Tensor<T>::operator[](size_t index) const {
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index];
}

template <typename T>
inline T& Tensor<T>::at(const std::vector<size_t>& indices) {
    return data_[calculate_flat_index(indices)];
}

template <typename T>
inline const T& Tensor<T>::at(const std::vector<size_t>& indices) const {
    return data_[calculate_flat_index(indices)];
}

template <typename T>
inline const std::vector<size_t>& Tensor<T>::shape() const { return dims_; }

template <typename T>
inline const std::vector<size_t>& Tensor<T>::strides() const { return strides_; }

template <typename T>
inline const std::vector<T>& Tensor<T>::data() const { return data_; }

template <typename T>
inline std::vector<T>& Tensor<T>::data() { return data_; }

template <typename T>
inline size_t Tensor<T>::size() const { return data_.size(); }

template <typename T>
inline size_t Tensor<T>::ndim() const { return dims_.size(); }

template <typename T>
inline const Device& Tensor<T>::device() const { return device_; }

template <typename T>
inline void Tensor<T>::to(Device device) {
    if (device_ == device) return;
    device_ = device;
    // TODO: Implement device transfer logic
}

template <typename T>
inline bool Tensor<T>::empty() const {
    return data_.empty();
}

template <typename T>
inline void Tensor<T>::reshape(const std::vector<size_t>& new_dims) {
    size_t new_size = std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<size_t>());
    if (new_size != data_.size()) {
        throw std::invalid_argument("New shape must have the same total size");
    }
    dims_ = new_dims;
    calculate_strides();
}

template <typename T>
inline void Tensor<T>::transform(const std::function<T(const T&)>& func) {
    std::transform(data_.begin(), data_.end(), data_.begin(), func);
}

template <typename T>
inline void Tensor<T>::permute(const std::vector<size_t>& axes) {
    if (axes.size() != dims_.size()) {
        throw std::invalid_argument("Number of axes must match tensor dimensions");
    }
    
    // Calculate new dimensions
    std::vector<size_t> new_dims(dims_.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] >= dims_.size()) {
            throw std::out_of_range("Axis index out of range");
        }
        new_dims[i] = dims_[axes[i]];
    }
    
    // Create a temporary copy of the data
    std::vector<T> new_data(data_.size());
    
    // Calculate new strides
    std::vector<size_t> new_strides(new_dims.size());
    if (!new_dims.empty()) {
        new_strides[new_dims.size() - 1] = 1;
        for (int i = new_dims.size() - 2; i >= 0; --i) {
            new_strides[i] = new_strides[i + 1] * new_dims[i + 1];
        }
    }
    
    // Permute the data
    std::vector<size_t> indices(dims_.size(), 0);
    for (size_t i = 0; i < data_.size(); ++i) {
        // Calculate the new index using the original strides
        size_t old_index = 0;
        for (size_t j = 0; j < dims_.size(); ++j) {
            old_index += indices[j] * strides_[j];
        }
        
        // Calculate the new index using the permuted dimensions
        size_t new_index = 0;
        for (size_t j = 0; j < axes.size(); ++j) {
            new_index += indices[axes[j]] * new_strides[j];
        }
        
        // Copy the data
        new_data[new_index] = data_[old_index];
        
        // Increment indices
        for (int j = dims_.size() - 1; j >= 0; --j) {
            indices[j]++;
            if (indices[j] < dims_[j]) {
                break;
            }
            indices[j] = 0;
        }
    }
    
    // Update the tensor state
    dims_ = std::move(new_dims);
    strides_ = std::move(new_strides);
    data_ = std::move(new_data);
}

template <typename T>
inline bool Tensor<T>::operator==(const Tensor<T>& other) const {
    return dims_ == other.dims_ && data_ == other.data_;
}

template <typename T>
inline bool Tensor<T>::operator!=(const Tensor<T>& other) const {
    return !(*this == other);
}

template <typename T>
inline std::string Tensor<T>::serialize() const {
    std::ostringstream oss;
    // Write dimensions
    oss << dims_.size() << " ";
    for (const auto& dim : dims_) {
        oss << dim << " ";
    }
    // Write data
    for (const auto& val : data_) {
        oss << val << " ";
    }
    return oss.str();
}

template <typename T>
inline Tensor<T> Tensor<T>::deserialize(const std::string& data) {
    std::istringstream iss(data);
    size_t ndims;
    iss >> ndims;
    
    std::vector<size_t> dims(ndims);
    for (size_t i = 0; i < ndims; ++i) {
        iss >> dims[i];
    }
    
    size_t total_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    std::vector<T> tensor_data(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        iss >> tensor_data[i];
    }
    
    return Tensor<T>(dims, tensor_data);
}

template <typename T>
inline void Tensor<T>::calculate_strides() {
    strides_.resize(dims_.size());
    if (dims_.empty()) return;
    
    strides_[dims_.size() - 1] = 1;
    for (int i = dims_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
}

template <typename T>
inline size_t Tensor<T>::calculate_flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != dims_.size()) {
        throw std::invalid_argument("Number of indices does not match tensor dimensions");
    }
    
    size_t flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= dims_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        flat_index += indices[i] * strides_[i];
    }
    return flat_index;
}

template <typename T>
inline void Tensor<T>::deserialize(const std::vector<uint8_t>& data_bytes) {
    std::string data_str(data_bytes.begin(), data_bytes.end());
    std::istringstream iss(data_str);
    size_t ndims;
    iss >> ndims;
    
    dims_.resize(ndims);
    for (size_t i = 0; i < ndims; ++i) {
        iss >> dims_[i];
    }
    
    size_t total_size = std::accumulate(dims_.begin(), dims_.end(), size_t{1}, std::multiplies<size_t>());
    data_.resize(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        T value;
        iss >> value;
        data_[i] = value;
    }
    
    calculate_strides();
}

template <typename T>
inline TensorView<T> Tensor<T>::view(size_t index) {
    if (index >= dims_[0]) {
        throw std::out_of_range("Index out of bounds");
    }
    std::vector<size_t> remaining_dims(dims_.begin() + 1, dims_.end());
    std::vector<size_t> remaining_strides(strides_.begin() + 1, strides_.end());
    return TensorView<T>(data_, index * strides_[0], remaining_dims, remaining_strides);
}

template <typename T>
inline const TensorView<T> Tensor<T>::view(size_t index) const {
    if (index >= dims_[0]) {
        throw std::out_of_range("Index out of bounds");
    }
    std::vector<size_t> remaining_dims(dims_.begin() + 1, dims_.end());
    std::vector<size_t> remaining_strides(strides_.begin() + 1, strides_.end());
    return TensorView<T>(data_, index * strides_[0], remaining_dims, remaining_strides);
}

// Explicit instantiations
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;

} // namespace dlf 