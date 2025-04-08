# Tutorial

## Getting Started

[... existing content ...]

## Using the Logging System

The framework includes a comprehensive logging system that can help you debug and monitor your deep learning applications.

### Basic Logging

First, initialize the logger in your application:

```cpp
#include <utils/logger.hpp>

int main() {
    // Initialize the logger with a log file
    dlf::Logger::getInstance().initialize("my_app.log");
    
    // Log a message
    DLF_LOG_INFO("Application started");
    
    // ... rest of your code ...
    
    return 0;
}
```

### Logging in Neural Networks

You can use the logger to monitor training progress and debug issues:

```cpp
class MyModel : public dlf::Model {
public:
    void forward(const dlf::Tensor& input) override {
        DLF_LOG_DEBUG("Input shape: {}", input.shape());
        
        // Forward pass
        auto x = layer1.forward(input);
        DLF_LOG_TRACE("Layer 1 output: {}", x);
        
        x = layer2.forward(x);
        DLF_LOG_TRACE("Layer 2 output: {}", x);
        
        // ... rest of forward pass ...
    }
    
    void backward(const dlf::Tensor& grad) override {
        DLF_LOG_DEBUG("Backward pass started");
        
        // Backward pass
        auto grad_x = layer2.backward(grad);
        DLF_LOG_TRACE("Layer 2 gradient: {}", grad_x);
        
        grad_x = layer1.backward(grad_x);
        DLF_LOG_TRACE("Layer 1 gradient: {}", grad_x);
        
        // ... rest of backward pass ...
    }
};
```

### Monitoring Training

Use the logger to track training progress:

```cpp
void train(Model& model, const Dataset& dataset, int epochs) {
    DLF_LOG_INFO("Starting training for {} epochs", epochs);
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        DLF_LOG_INFO("Epoch {}/{}", epoch + 1, epochs);
        
        for (const auto& batch : dataset) {
            // Forward pass
            auto output = model.forward(batch.input);
            DLF_LOG_TRACE("Batch output: {}", output);
            
            // Compute loss
            auto loss = compute_loss(output, batch.target);
            DLF_LOG_DEBUG("Batch loss: {}", loss);
            
            // Backward pass
            model.backward(loss.gradient());
            
            // Update weights
            optimizer.step();
        }
        
        // Log epoch statistics
        DLF_LOG_INFO("Epoch {} complete. Average loss: {}", 
                    epoch + 1, 
                    compute_average_loss(dataset));
    }
    
    DLF_LOG_INFO("Training complete");
}
```

### Error Handling

Use the logger for error handling and debugging:

```cpp
Tensor load_model(const std::string& path) {
    try {
        DLF_LOG_INFO("Loading model from {}", path);
        // ... load model ...
        return model;
    } catch (const std::exception& e) {
        DLF_LOG_ERROR("Failed to load model: {}", e.what());
        throw;
    }
}

void process_data(const Tensor& data) {
    if (data.empty()) {
        DLF_LOG_WARN("Empty input tensor");
        return;
    }
    
    if (!data.is_valid()) {
        DLF_LOG_ERROR("Invalid tensor data");
        throw std::runtime_error("Invalid tensor data");
    }
    
    // ... process data ...
}
```

### Log Levels

Use different log levels appropriately:

- `TRACE`: Detailed debugging information
- `DEBUG`: General debugging information
- `INFO`: General information about program execution
- `WARN`: Warning messages for potentially problematic situations
- `ERROR`: Error messages for recoverable errors
- `CRITICAL`: Critical errors that may lead to program termination

### Best Practices

1. Initialize the logger early in your application
2. Use appropriate log levels
3. Include relevant context in log messages
4. Use string formatting for complex messages
5. Flush logs before critical operations
6. Use source location for debugging

Example:

```cpp
void critical_operation() {
    DLF_LOG_INFO("Starting critical operation");
    
    try {
        // ... perform operation ...
        DLF_LOG_INFO("Operation completed successfully");
    } catch (const std::exception& e) {
        DLF_LOG_ERROR("Operation failed: {}", e.what());
        // Ensure logs are written before throwing
        dlf::Logger::getInstance().flush();
        throw;
    }
}
```

[... rest of the tutorial ...] 