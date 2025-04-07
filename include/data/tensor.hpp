#pragma once

#include <armadillo>
// #include <glog/logging.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace dlf {

template <typename T, std::size_t N, std::size_t Dim>
struct TensorRef {

public:
  T *data_;
  const std::vector<size_t> &shape_;
  std::size_t offset_;

  explicit TensorRef(T *data, const std::vector<size_t> &shape, std::size_t offset = 0)
      : data_(data)
      , shape_(shape)
      , offset_(offset)
  {}

  // Access operator recursively
  auto operator[](std::size_t idx) {
    // Calculate stride for current dimension
    std::size_t stride = 1;
    for (std::size_t i = Dim + 1; i < shape_.size(); ++i) {
      stride *= shape_[i];
    }
    std::size_t new_offset = offset_ + idx * stride;

    if constexpr (Dim + 1 < N) {
      return TensorRef<T, N, Dim + 1>(data_, shape_, new_offset);
    } else {
      // The last dimension, return the value
      return data_[new_offset];
    }
  }

};

template <typename T, std::size_t N = 1>
class Tensor {

  std::shared_ptr<T[]> data_;
  std::vector<size_t> shape_;

public:
  Tensor() : data_(nullptr), shape_({0}) {}

  explicit Tensor(const std::vector<size_t> &shape) : shape_(shape) {
    size_t size =
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
    data_ = std::make_shared<T[]>(size);
  }

  Tensor(const std::vector<size_t> &shape, const T &value) : shape_(shape) {
    size_t size =
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
    data_ = std::make_shared<T[]>(size);
    std::cout << "size: " << size << std::endl;
    std::fill(data_.get(), data_.get() + size, value);
  }

  Tensor(const std::vector<size_t> &shape, const std::vector<T> &data)
      : shape_(shape) {
    size_t size =
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
    if (data.size() != size) {
      // LOG(FATAL) << "Invalid data size: " << data.size() << " vs " << size;
      throw std::runtime_error("Invalid data size");
    }
    data_ = std::make_shared<T[]>(size);
    std::copy(data.begin(), data.end(), data_.get());
  }

  Tensor(const std::vector<size_t> &shape, const T *data) : shape_(shape) {
    size_t size =
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
    data_ = std::make_shared<T[]>(size);
    std::copy(data, data + size, data_.get());
  }

  Tensor(const Tensor<T, N> &other) : shape_(other.shape_) {
    size_t size =
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
    data_ = std::make_shared<T[]>(size);
    std::copy(other.data_.get(), other.data_.get() + size, data_.get());
  }

  Tensor(Tensor<T, N> &&other) noexcept
      : shape_(other.shape_), data_(std::move(other.data_)) {}

  Tensor<T, N> &operator=(const Tensor<T, N> &other) {
    if (this != &other) {
      shape_ = other.shape_;
      size_t size =
          std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = std::make_shared<T[]>(size);
      std::copy(other.data_.get(), other.data_.get() + size, data_.get());
    }
    return *this;
  }

  Tensor<T, N> &operator=(Tensor<T, N> &&other) noexcept {
    if (this != &other) {
      shape_ = other.shape_;
      data_ = std::move(other.data_);
    }
    return *this;
  }

  [[nodiscard]] const std::vector<size_t> &shape() const { return shape_; }

  [[nodiscard]] size_t size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<>());
  }

  const T *data() const { return data_.get(); }

  T *data() { return data_.get(); }

  auto operator[](std::size_t idx) {
    return TensorRef<T, N, 1>(data_.get(), shape_, idx * stride(0));
  }

  auto operator[](std::size_t idx) const {
    return TensorRef<T, N, 1>(data_.get(), shape_, idx * stride(0));
  }

  template <typename U> Tensor<U> cast() const {
    Tensor<U> result(shape_);
    std::copy(data_.get(), data_.get() + size(), result.data_.get());
    return result;
  }

  template <typename U> Tensor<U> as_type() const { return cast<U>(); }

  template <typename U> Tensor<U> as_type() { return cast<U>(); }

  [[nodiscard]] std::size_t bytes() const {
    return sizeof(T) * std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
  }

  [[nodiscard]] std::size_t stride(size_t dim = 0) const {
    if (dim >= shape_.size()) {
      throw std::out_of_range("Dimension out of range");
    }

    std::vector<std::size_t> strides_info(shape_.size());
    strides_info.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
      strides_info[i] = strides_info[i + 1] * shape_[i + 1];
    }

    return strides_info[dim];
  }

  [[nodiscard]] bool empty() const { return size() == 0; }

  [[nodiscard]] std::vector<std::size_t> strides() const {
    std::vector<std::size_t> strides(shape_.size());
    strides.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape_[i + 1];
    }
    return strides;
  }

  [[nodiscard]] std::vector<std::size_t> permute(const std::vector<size_t> &order) const {
    std::vector<std::size_t> new_shape(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i) {
      new_shape[i] = shape_[order[i]];
    }
    return new_shape;
  }

  void reshape(const std::vector<size_t> &new_shape) {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<>());
    if (new_size != size()) {
      throw std::runtime_error("Invalid reshape: incompatible sizes");
    }
    shape_ = new_shape;
  }

  void transform(const std::function<T(const T &)> &func) {
    size_t size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
    for (size_t i = 0; i < size; ++i) {
      auto res = func(data_[i]);
      data_[i] = func(data_[i]);
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const Tensor<T, N> &tensor) {
    os << "Tensor(";
    for (size_t i = 0; i < tensor.shape_.size(); ++i) {
      os << tensor.shape_[i];
      if (i < tensor.shape_.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
    return os;
  }
};

template <typename T>
using Vector = Tensor<T, 1>;

template <typename T>
using Matrix = Tensor<T, 2>;

template <typename T>
using Cube = Tensor<T, 3>;

} // namespace dlf