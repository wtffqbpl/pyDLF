#pragma once

#include <vector>
#include <cstddef>

namespace dlf {

template <typename T>
class TensorView {
public:
    TensorView(std::vector<T>& data, size_t offset, const std::vector<size_t>& dims, const std::vector<size_t>& strides);
    TensorView(const std::vector<T>& data, size_t offset, const std::vector<size_t>& dims, const std::vector<size_t>& strides);
    TensorView(const TensorView& other);
    TensorView& operator=(const TensorView& other);
    TensorView& operator=(T value);

    bool operator==(const TensorView& other) const;
    bool operator!=(const TensorView& other) const;
    bool operator==(T value) const;
    bool operator!=(T value) const;
    operator T() const;

    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;
    T& value();
    const T& value() const;
    void set_value(const T& value);

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    size_t size() const;
    size_t ndim() const;

    TensorView<T> view(size_t index);
    const TensorView<T> view(size_t index) const;

protected:
    size_t calculate_flat_index(const std::vector<size_t>& indices) const;

private:
    std::vector<T>& data_;
    size_t offset_;
    std::vector<size_t> dims_;
    std::vector<size_t> strides_;
    T value_;
    bool is_const_;
};

// Forward declare the bool specialization
template<>
class TensorView<bool>;

}  // namespace dlf

#include "tensor_view.hpp"