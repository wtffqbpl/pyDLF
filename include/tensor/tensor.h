#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <functional>
#include "device.h"
#include "tensor_view.h"

namespace dlf {

// Forward declarations
template <typename T>
class Tensor;
template <typename T>
class TensorView;

// Specialization for bool to handle std::vector<bool> reference issues
template<>
class Tensor<bool> {
public:
    Tensor(const std::vector<size_t>& dims);
    Tensor(const std::vector<size_t>& dims, const std::vector<bool>& data);
    Tensor(const std::vector<size_t>& dims, const Device& device);
    Tensor(const std::vector<size_t>& dims, bool value);

    bool operator[](size_t index) const;
    void set(size_t index, bool value);
    bool at(const std::vector<size_t>& indices) const;
    void set_at(const std::vector<size_t>& indices, bool value);

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    const std::vector<bool>& data() const;
    size_t size() const;
    size_t ndim() const;
    const Device& device() const;
    bool empty() const;

    void reshape(const std::vector<size_t>& new_dims);
    void transform(const std::function<bool(bool)>& func);
    void permute(const std::vector<size_t>& axes);

    bool operator==(const Tensor<bool>& other) const;
    bool operator!=(const Tensor<bool>& other) const;

    void to(Device device);
    std::string serialize() const;
    static Tensor<bool> deserialize(const std::string& data);
    void deserialize(const std::vector<uint8_t>& data);

    TensorView<bool> view(size_t index);
    const TensorView<bool> view(size_t index) const;

protected:
    void calculate_strides();
    size_t calculate_flat_index(const std::vector<size_t>& indices) const;

private:
    std::vector<size_t> dims_;
    std::vector<size_t> strides_;
    std::vector<bool> data_;
    Device device_;
};

// General template
template <typename T>
class Tensor {
public:
    Tensor(const std::vector<size_t>& dims);
    Tensor(const std::vector<size_t>& dims, const std::vector<T>& data);
    Tensor(const std::vector<size_t>& dims, const Device& device);
    Tensor(const std::vector<size_t>& dims, T value);

    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    const std::vector<T>& data() const;
    std::vector<T>& data();
    size_t size() const;
    size_t ndim() const;
    const Device& device() const;
    bool empty() const;

    void reshape(const std::vector<size_t>& new_dims);
    void transform(const std::function<T(const T&)>& func);
    void permute(const std::vector<size_t>& axes);

    bool operator==(const Tensor<T>& other) const;
    bool operator!=(const Tensor<T>& other) const;

    void to(Device device);
    std::string serialize() const;
    static Tensor<T> deserialize(const std::string& data);
    void deserialize(const std::vector<uint8_t>& data);

    TensorView<T> view(size_t index);
    const TensorView<T> view(size_t index) const;

protected:
    void calculate_strides();
    size_t calculate_flat_index(const std::vector<size_t>& indices) const;

private:
    std::vector<size_t> dims_;
    std::vector<size_t> strides_;
    std::vector<T> data_;
    Device device_;
};

} // namespace dlf

#include "tensor_impl.hpp"
