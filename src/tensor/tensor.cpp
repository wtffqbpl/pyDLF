#include "tensor/tensor.h"
#include "tensor/tensor.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace dlf
{

template<>
Tensor<float>::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
    : shape_(shape),
      data_(data.empty() ? std::vector<float>(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())) : data)
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

template<>
Tensor<float>::Tensor(const std::vector<size_t>& shape, const float& value)
    : shape_(shape),
      data_(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), value)
{
    if (shape.empty())
    {
        throw std::invalid_argument("Shape cannot be empty");
    }
}

template<>
const std::vector<size_t>& Tensor<float>::shape() const
{
    return shape_;
}

template<>
std::vector<float>& Tensor<float>::data()
{
    return data_;
}

template<>
const std::vector<float>& Tensor<float>::data() const
{
    return data_;
}

template<>
float& Tensor<float>::at(const std::vector<size_t>& indices)
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

template<>
const float& Tensor<float>::at(const std::vector<size_t>& indices) const
{
    return const_cast<Tensor<float>*>(this)->at(indices);
}

template<>
TensorView<float> Tensor<float>::view(size_t index)
{
    if (index >= shape_[0])
    {
        throw std::out_of_range("Index out of range");
    }
    return TensorView<float>(*this, index);
}

template<>
const TensorView<float> Tensor<float>::view(size_t index) const
{
    return const_cast<Tensor<float>*>(this)->view(index);
}

// Template instantiations
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;

template class TensorView<float>;
template class TensorView<double>;
template class TensorView<int>;

}  // namespace dlf