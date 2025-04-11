#include "tensor/tensor.h"
#include "utils/logger.hpp"
#include <vector>

void dummy_func()
{
    // Initialize logger
    dlf::Logger::getInstance().initialize();

    // Example logging
    DLF_LOG_INFO("Initializing DLF framework");
    DLF_LOG_DEBUG("Debug information");
    DLF_LOG_WARN("Warning message");
    DLF_LOG_ERROR("Error message");

    // Example with tensor
    std::vector<int> data(6, 1);  // 6 elements (2*3) all initialized to 1
    dlf::Tensor<int> tensor({2, 3}, data);
    DLF_LOG_INFO("Created tensor with shape: {}x{}", tensor.shape()[0], tensor.shape()[1]);
}

int main() {
    // Create a tensor with dimensions {2, 3} and initialize all elements to 1
    std::vector<int> data(6, 1);  // 6 elements (2*3) all initialized to 1
    dlf::Tensor<int> tensor({2, 3}, data);
    return 0;
}