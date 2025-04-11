#pragma once

#include <string>

namespace dlf {

enum class DeviceType {
    CPU,
    CUDA  // 未来支持
};

class Device {
public:
    Device() : type_(DeviceType::CPU), index_(0) {}
    explicit Device(DeviceType type, int index = 0) : type_(type), index_(index) {}
    
    static Device cpu() { return Device(DeviceType::CPU); }
    static Device cuda(int index = 0) { return Device(DeviceType::CUDA, index); }
    
    DeviceType type() const { return type_; }
    int index() const { return index_; }
    
    bool is_cpu() const { return type_ == DeviceType::CPU; }
    bool is_cuda() const { return type_ == DeviceType::CUDA; }
    
    std::string str() const {
        if (is_cpu()) {
            return "cpu";
        } else if (is_cuda()) {
            return "cuda:" + std::to_string(index_);
        }
        return "unknown";
    }
    
    bool operator==(const Device& other) const {
        return type_ == other.type_ && index_ == other.index_;
    }
    
    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        if (type_ == DeviceType::CPU) {
            return "cpu";
        } else {
            return "cuda:" + std::to_string(index_);
        }
    }

private:
    DeviceType type_;
    int index_;
};

} // namespace dlf 