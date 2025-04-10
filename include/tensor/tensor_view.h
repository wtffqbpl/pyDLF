#pragma once

#include <vector>
#include "tensor/tensor.h"

namespace dlf {

template<typename T>
class TensorView {
public:
    // Constructors
    TensorView(Tensor<T>& tensor, size_t index);
    TensorView(const Tensor<T>& tensor, size_t index);
    TensorView(Tensor<T>& tensor, const std::vector<size_t>& indices);
    TensorView(const Tensor<T>& tensor, const std::vector<size_t>& indices);
    TensorView(const TensorView& other) = default;

    // Access methods
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;
    void set_at(const std::vector<size_t>& indices, T value);
    TensorView view(size_t index) const;
    T& value();
    const T& value() const;
    const std::vector<size_t>& remaining_dims() const;

    // Operators
    TensorView& operator=(T value);
    bool operator==(const TensorView& other) const;
    bool operator!=(const TensorView& other) const;
    bool operator==(T value) const;
    bool operator!=(T value) const;

    // Conversion operator
    operator T() const {
        if (!remaining_dims_.empty()) {
            throw std::runtime_error("Cannot convert non-scalar tensor view to value");
        }
        return value();
    }

private:
    Tensor<T>& tensor_;
    std::vector<size_t> indices_;
    std::vector<size_t> remaining_dims_;
};

} // namespace dlf

#include "tensor/tensor_view.hpp" 