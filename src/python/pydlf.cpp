#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tensor/tensor.h"

namespace py = pybind11;
using namespace dlf;

PYBIND11_MODULE(_pydlf, m) {
    py::class_<Tensor<float>>(m, "Tensor", py::module_local())
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, float>())
        .def("shape", &Tensor<float>::shape)
        .def("data", static_cast<const std::vector<float>& (Tensor<float>::*)() const>(&Tensor<float>::data))
        .def("view", static_cast<TensorView<float> (Tensor<float>::*)(size_t)>(&Tensor<float>::view))
        .def("at", static_cast<float& (Tensor<float>::*)(const std::vector<size_t>&)>(&Tensor<float>::at))
        .def("set_at", [](Tensor<float>& self, const std::vector<size_t>& indices, float value) {
            self.at(indices) = value;
        })
        .def("to_numpy", [](Tensor<float>& self) {
            // Create a copy of the data to ensure memory safety
            std::vector<float> data_copy = self.data();
            float* data_ptr = new float[data_copy.size()];
            std::copy(data_copy.begin(), data_copy.end(), data_ptr);
            
            auto shape = self.shape();
            std::vector<ssize_t> numpy_shape(shape.begin(), shape.end());
            std::vector<ssize_t> strides(shape.size());
            ssize_t stride = sizeof(float);
            for (int i = shape.size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            return py::array_t<float>(
                numpy_shape,
                strides,
                data_ptr,
                py::capsule(data_ptr, [](void* ptr) { delete[] static_cast<float*>(ptr); })
            );
        });

    py::class_<TensorView<float>>(m, "TensorView", py::module_local())
        .def("view", static_cast<TensorView<float> (TensorView<float>::*)(size_t)>(&TensorView<float>::view))
        .def("at", static_cast<float& (TensorView<float>::*)(const std::vector<size_t>&)>(&TensorView<float>::at))
        .def("set_at", [](TensorView<float>& self, const std::vector<size_t>& indices, float value) {
            self.at(indices) = value;
        })
        .def("value", static_cast<float& (TensorView<float>::*)()>(&TensorView<float>::value))
        .def("__float__", [](const TensorView<float>& self) { return static_cast<float>(self); })
        .def("to_numpy", [](TensorView<float>& self) {
            // Get the shape of the view
            std::vector<size_t> shape = self.shape();
            std::vector<ssize_t> numpy_shape(shape.begin(), shape.end());
            
            // Create a copy of the data
            std::vector<float> data_copy;
            data_copy.reserve(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()));
            
            // Recursively fill the data array
            std::function<void(const TensorView<float>&, const std::vector<size_t>&)> fill_data = 
                [&](const TensorView<float>& view, const std::vector<size_t>& curr_indices) {
                    if (curr_indices.size() == shape.size()) {
                        data_copy.push_back(view.at(curr_indices));
                        return;
                    }
                    
                    size_t dim = curr_indices.size();
                    std::vector<size_t> next_indices = curr_indices;
                    next_indices.push_back(0);
                    for (size_t i = 0; i < shape[dim]; ++i) {
                        next_indices.back() = i;
                        fill_data(view, next_indices);
                    }
                };
            
            fill_data(self, std::vector<size_t>());
            
            // Create numpy array
            float* data_ptr = new float[data_copy.size()];
            std::copy(data_copy.begin(), data_copy.end(), data_ptr);
            
            std::vector<ssize_t> strides(shape.size());
            ssize_t stride = sizeof(float);
            for (int i = shape.size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            
            return py::array_t<float>(
                numpy_shape,
                strides,
                data_ptr,
                py::capsule(data_ptr, [](void* ptr) { delete[] static_cast<float*>(ptr); })
            );
        });
}