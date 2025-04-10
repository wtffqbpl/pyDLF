#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tensor/tensor.h"
#include "utils/logger.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_dlf, m) {
    m.doc() = "Python bindings for the dlf C++ library"; // optional module docstring

    // Tensor class
    py::class_<dlf::Tensor<float>>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&>())
        .def(py::init<const std::vector<size_t>&, const float&>())
        .def("shape", &dlf::Tensor<float>::shape)
        .def("data", [](dlf::Tensor<float>& t) -> std::vector<float>& {
            return t.data();
        })
        .def("size", &dlf::Tensor<float>::size)
        .def("empty", &dlf::Tensor<float>::empty)
        .def("strides", &dlf::Tensor<float>::strides)
        .def("reshape", &dlf::Tensor<float>::reshape)
        .def("transform", &dlf::Tensor<float>::transform)
        .def("permute", &dlf::Tensor<float>::permute)
        .def("at", [](dlf::Tensor<float>& t, const std::vector<size_t>& indices) -> float& {
            return t.at(indices);
        })
        .def("set_at", [](dlf::Tensor<float>& t, const std::vector<size_t>& indices, float value) {
            t.at(indices) = value;
        })
        .def("view", [](dlf::Tensor<float>& t, size_t index) -> dlf::TensorView<float> {
            return t.view(index);
        })
        .def("to_numpy", [](const dlf::Tensor<float>& t) {
            auto shape = t.shape();
            auto data = t.data();
            return py::array_t<float>(shape, data.data());
        });

    // TensorView class
    py::class_<dlf::TensorView<float>>(m, "TensorView")
        .def("at", [](dlf::TensorView<float>& v, const std::vector<size_t>& indices) -> float& {
            return v.at(indices);
        })
        .def("set_at", [](dlf::TensorView<float>& v, const std::vector<size_t>& indices, float value) {
            v.at(indices) = value;
        })
        .def("view", [](dlf::TensorView<float>& v, size_t index) -> dlf::TensorView<float> {
            return v.view(index);
        })
        .def("value", [](dlf::TensorView<float>& v) -> float& {
            return v.value();
        })
        .def("remaining_dims", &dlf::TensorView<float>::remaining_dims);
}