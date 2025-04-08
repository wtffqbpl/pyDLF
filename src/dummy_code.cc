#include <data/tensor.hpp>
#include <utils/logger.hpp>

void dummy_func() {
    // Initialize logger
    dlf::Logger::getInstance().initialize();
    
    // Example logging
    DLF_LOG_INFO("Initializing DLF framework");
    DLF_LOG_DEBUG("Debug information");
    DLF_LOG_WARN("Warning message");
    DLF_LOG_ERROR("Error message");
    
    // Example with tensor
    dlf::Matrix<int> tensor({2, 3}, 1);
    DLF_LOG_INFO("Created tensor with shape: {}x{}", tensor.shape()[0], tensor.shape()[1]);
}