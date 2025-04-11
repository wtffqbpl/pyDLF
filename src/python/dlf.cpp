#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <numeric>
#include <typeinfo>

#include "tensor/tensor.h"
#include "tensor/tensor_view.h"
#include "tensor/device.h"

namespace py = pybind11;

// Helper for getting the typename as a string
template <typename T>
std::string type_name() {
    if (std::is_same<T, float>::value) return "float";
    if (std::is_same<T, double>::value) return "double";
    if (std::is_same<T, int>::value) return "int";
    if (std::is_same<T, long>::value) return "long";
    if (std::is_same<T, bool>::value) return "bool";
    return "unknown";
}

// Register Device class and DeviceType enum
void register_device(py::module& m) {
    py::enum_<dlf::DeviceType>(m, "DeviceType")
        .value("CPU", dlf::DeviceType::CPU)
        .value("CUDA", dlf::DeviceType::CUDA)
        .export_values();

    py::class_<dlf::Device>(m, "Device")
        .def(py::init<>())
        .def_static("cpu", &dlf::Device::cpu)
        .def_static("cuda", &dlf::Device::cuda)
        .def("type", &dlf::Device::type)
        .def("index", &dlf::Device::index)
        .def("is_cpu", &dlf::Device::is_cpu)
        .def("is_cuda", &dlf::Device::is_cuda)
        .def("__str__", &dlf::Device::str);
}

// Template function to register tensor for different types
template <typename T>
void register_tensor(py::module& m, const char* name) {
    auto cls = py::class_<dlf::Tensor<T>>(m, name)
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, const std::vector<T>&>())
        .def(py::init<const std::vector<size_t>&, const dlf::Device&>())
        .def("shape", &dlf::Tensor<T>::shape)
        .def("strides", &dlf::Tensor<T>::strides)
        .def("data", static_cast<const std::vector<T>& (dlf::Tensor<T>::*)() const>(&dlf::Tensor<T>::data))
        .def("device", &dlf::Tensor<T>::device)
        .def("size", &dlf::Tensor<T>::size)
        .def("empty", &dlf::Tensor<T>::empty)
        .def("ndim", &dlf::Tensor<T>::ndim)
        .def("reshape", &dlf::Tensor<T>::reshape)
        .def("transform", [](dlf::Tensor<T>& t, const py::function& func) {
            t.transform([&func](const T& x) {
                return py::cast<T>(func(x));
            });
        })
        .def("permute", &dlf::Tensor<T>::permute)
        .def("view", static_cast<dlf::TensorView<T> (dlf::Tensor<T>::*)(size_t)>(&dlf::Tensor<T>::view))
        .def("to", [](dlf::Tensor<T>& t, const std::string& device_str) {
            dlf::Device device = dlf::Device::cpu();
            if (device_str == "cuda" || device_str.substr(0, 5) == "cuda:") {
                int device_idx = 0;
                if (device_str.length() > 5) {
                    device_idx = std::stoi(device_str.substr(5));
                }
                device = dlf::Device::cuda(device_idx);
            }
            t.to(device);
            return t;
        })
        .def("at", [](dlf::Tensor<T>& t, const std::vector<size_t>& indices) -> T& { return t.at(indices); })
        .def("set_at", [](dlf::Tensor<T>& t, const std::vector<size_t>& indices, const T& value) { t.at(indices) = value; })
        .def("to_numpy", [](dlf::Tensor<T>& t) {
            std::vector<size_t> shape = t.shape();
            std::vector<size_t> strides = t.strides();
            // Convert strides from elements to bytes
            std::vector<size_t> numpy_strides;
            for (auto s : strides) {
                numpy_strides.push_back(s * sizeof(T));
            }
            py::array_t<T> result(shape, numpy_strides);
            auto buf = result.request();
            T* ptr = static_cast<T*>(buf.ptr);
            const std::vector<T>& data = t.data();
            std::copy(data.begin(), data.end(), ptr);
            return result;
        });
}

// Template specialization for bool
template <>
void register_tensor<bool>(py::module& m, const char* name) {
    auto cls = py::class_<dlf::Tensor<bool>>(m, name)
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, const std::vector<bool>&>())
        .def(py::init<const std::vector<size_t>&, const dlf::Device&>())
        .def("shape", &dlf::Tensor<bool>::shape)
        .def("strides", &dlf::Tensor<bool>::strides)
        .def("data", static_cast<const std::vector<bool>& (dlf::Tensor<bool>::*)() const>(&dlf::Tensor<bool>::data))
        .def("device", &dlf::Tensor<bool>::device)
        .def("size", &dlf::Tensor<bool>::size)
        .def("empty", &dlf::Tensor<bool>::empty)
        .def("ndim", &dlf::Tensor<bool>::ndim)
        .def("reshape", &dlf::Tensor<bool>::reshape)
        .def("transform", [](dlf::Tensor<bool>& t, const py::function& func) {
            t.transform([&func](bool x) {
                return py::cast<bool>(func(x));
            });
        })
        .def("permute", &dlf::Tensor<bool>::permute)
        .def("view", static_cast<dlf::TensorView<bool> (dlf::Tensor<bool>::*)(size_t)>(&dlf::Tensor<bool>::view))
        .def("to", [](dlf::Tensor<bool>& t, const std::string& device_str) {
            dlf::Device device = dlf::Device::cpu();
            if (device_str == "cuda" || device_str.substr(0, 5) == "cuda:") {
                int device_idx = 0;
                if (device_str.length() > 5) {
                    device_idx = std::stoi(device_str.substr(5));
                }
                device = dlf::Device::cuda(device_idx);
            }
            t.to(device);
            return t;
        })
        .def("at", [](dlf::Tensor<bool>& t, const std::vector<size_t>& indices) { return t.at(indices); })
        .def("set_at", [](dlf::Tensor<bool>& t, const std::vector<size_t>& indices, bool value) { 
            std::vector<bool>& data = const_cast<std::vector<bool>&>(t.data());
            size_t idx = 0;
            const auto& shape = t.shape();
            const auto& strides = t.strides();
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] >= shape[i]) {
                    throw std::out_of_range("Index out of range");
                }
                idx += indices[i] * strides[i];
            }
            data[idx] = value;
        })
        .def("to_numpy", [](dlf::Tensor<bool>& t) {
            std::vector<size_t> shape = t.shape();
            std::vector<size_t> strides = t.strides();
            // Convert strides from elements to bytes
            std::vector<size_t> numpy_strides;
            for (auto s : strides) {
                numpy_strides.push_back(s * sizeof(bool));
            }
            py::array_t<bool> result(shape, numpy_strides);
            auto buf = result.request();
            bool* ptr = static_cast<bool*>(buf.ptr);
            const std::vector<bool>& data = t.data();
            std::copy(data.begin(), data.end(), ptr);
            return result;
        });
}

// Template function to register tensor view for different types
template <typename T>
void register_tensor_view(py::module& m, const char* name) {
    auto cls = py::class_<dlf::TensorView<T>>(m, name)
        .def("shape", &dlf::TensorView<T>::shape)
        .def("strides", &dlf::TensorView<T>::strides)
        .def("size", &dlf::TensorView<T>::size)
        .def("ndim", &dlf::TensorView<T>::ndim)
        .def("at", [](dlf::TensorView<T>& v, const std::vector<size_t>& indices) -> T& { return v.at(indices); })
        .def("set_at", [](dlf::TensorView<T>& v, const std::vector<size_t>& indices, const T& value) { v.at(indices) = value; })
        .def("value", [](dlf::TensorView<T>& v) -> T& { return v.value(); })
        .def("set_value", &dlf::TensorView<T>::set_value)
        .def("remaining_dims", [](dlf::TensorView<T>& v) { return v.shape(); });
}

// Template specialization for bool
template <>
void register_tensor_view<bool>(py::module& m, const char* name) {
    auto cls = py::class_<dlf::TensorView<bool>>(m, name)
        .def("shape", &dlf::TensorView<bool>::shape)
        .def("strides", &dlf::TensorView<bool>::strides)
        .def("size", &dlf::TensorView<bool>::size)
        .def("ndim", &dlf::TensorView<bool>::ndim)
        .def("at", [](dlf::TensorView<bool>& v, const std::vector<size_t>& indices) { return v.at(indices); })
        .def("set_at", [](dlf::TensorView<bool>& v, const std::vector<size_t>& indices, bool value) { 
            size_t idx = 0;
            const auto& shape = v.shape();
            const auto& strides = v.strides();
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] >= shape[i]) {
                    throw std::out_of_range("Index out of range");
                }
                idx += indices[i] * strides[i];
            }
            v.set_value(value);
        })
        .def("value", [](dlf::TensorView<bool>& v) { return v.value(); })
        .def("set_value", &dlf::TensorView<bool>::set_value)
        .def("remaining_dims", [](dlf::TensorView<bool>& v) { return v.shape(); });
}

PYBIND11_MODULE(_pydlf, m) {
    m.doc() = "Python bindings for dlf";

    // Register Device class and DeviceType enum
    register_device(m);

    // Register Tensor for different types
    register_tensor<float>(m, "Tensor");
    register_tensor<double>(m, "TensorDouble");
    register_tensor<int>(m, "TensorInt");
    register_tensor<long>(m, "TensorLong");
    register_tensor<bool>(m, "TensorBool");

    // Register TensorView for different types
    register_tensor_view<float>(m, "TensorView");
    register_tensor_view<double>(m, "TensorViewDouble");
    register_tensor_view<int>(m, "TensorViewInt");
    register_tensor_view<long>(m, "TensorViewLong");
    register_tensor_view<bool>(m, "TensorViewBool");
}