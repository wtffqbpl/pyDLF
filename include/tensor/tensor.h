#pragma once

#include <array>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dlf
{

// Forward declaration for TensorView
template <typename T>
class TensorView;

template <typename T>
class Tensor
{
public:
    // Constructors
    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data = {});
    Tensor(const std::vector<size_t>& shape, const T& value);

    // Copy and move constructors/assignments
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Getters
    const std::vector<size_t>& shape() const
    {
        return shape_;
    }
    const std::vector<T>& data() const
    {
        return data_;
    }
    std::vector<T>& data()
    {
        return data_;
    }
    size_t size() const
    {
        return data_.size();
    }
    bool empty() const
    {
        return data_.empty();
    }

    // Strides calculation
    std::vector<size_t> strides() const;

    // Element access
    T& operator[](size_t index);
    const T& operator[](size_t index) const;

    // Multi-dimensional access
    TensorView<T> view(size_t index);
    const TensorView<T> view(size_t index) const;

    // Convenience method for at()
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;

    // Function call operator for multi-dimensional access
    template <typename... Args>
    T& operator()(Args... indices);

    template <typename... Args>
    const T& operator()(Args... indices) const;

    // Reshape operations
    void reshape(const std::vector<size_t>& new_shape);

    // Transform operation
    void transform(std::function<T(const T&)> func);

    // Permute operation
    std::vector<size_t> permute(const std::vector<size_t>& permutation);

    // Comparison operators
    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    // Serialization
    std::string serialize() const;
    static Tensor deserialize(const std::string& str);

private:
    std::vector<size_t> shape_;
    std::vector<T> data_;

    size_t calculate_size(const std::vector<size_t>& shape) const;
    void validate_shape(const std::vector<size_t>& shape) const;
    size_t calculate_index(const std::vector<size_t>& indices) const;
    void validate_indices(const std::vector<size_t>& indices) const;

    // Friend declaration for TensorView
    friend class TensorView<T>;
};

// TensorView class for nested operator[] access
template <typename T>
class TensorView
{
public:
    // Constructors
    TensorView(Tensor<T>& tensor, size_t index);
    TensorView(const Tensor<T>& tensor, size_t index);
    TensorView(Tensor<T>& tensor, const std::vector<size_t>& indices);
    TensorView(const Tensor<T>& tensor, const std::vector<size_t>& indices);
    TensorView(const TensorView& other) : tensor_(other.tensor_), indices_(other.indices_)
    {
    }

    // Access methods
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;
    TensorView<T> view(size_t index);
    const TensorView<T> view(size_t index) const;

    // Operators
    TensorView<T> operator[](size_t index);
    const TensorView<T> operator[](size_t index) const;

    // Value access
    T& value()
    {
        return tensor_.at(indices_);
    }
    const T& value() const
    {
        return tensor_.at(indices_);
    }

    // Implicit conversion to T
    operator T() const
    {
        return value();
    }

    // Assignment operators
    TensorView& operator=(const T& value)
    {
        if (is_const_)
        {
            throw std::runtime_error("Cannot modify const tensor view");
        }
        tensor_.at(indices_) = value;
        return *this;
    }

    TensorView& operator=(const TensorView& other)
    {
        if (is_const_)
        {
            throw std::runtime_error("Cannot modify const tensor view");
        }
        tensor_.at(indices_) = other.value();
        return *this;
    }

    // Comparison operators
    bool operator==(const T& other) const
    {
        return value() == other;
    }
    bool operator!=(const T& other) const
    {
        return value() != other;
    }
    bool operator==(const TensorView& other) const
    {
        return value() == other.value();
    }
    bool operator!=(const TensorView& other) const
    {
        return value() != other.value();
    }

    // Getters
    const std::vector<size_t>& shape() const
    {
        return tensor_.shape();
    }
    size_t remaining_dims() const
    {
        return tensor_.shape().size() - indices_.size();
    }

private:
    Tensor<T>& tensor_;
    std::vector<size_t> indices_;
    std::vector<size_t> remaining_shape_;
    bool is_const_ = false;

    void validate_index(size_t index) const;
    size_t calculate_index() const;
};

} // namespace dlf

// Template implementation
#include "tensor/tensor.hpp"
