---
layout: default
title: API Reference
---

# DLF API Reference

## Core Modules

### dlf.tensor

The tensor module provides the basic data structure for DLF.

```python
import dlf

# Create a tensor
x = dlf.tensor([1, 2, 3, 4])
y = dlf.tensor([[1, 2], [3, 4]], device="cuda")

# Common operations
z = x + y
w = dlf.matmul(x, y)
```

### dlf.nn

The neural network module contains building blocks for creating neural networks.

```python
import dlf

# Create network layers
linear = dlf.nn.Linear(784, 256)
relu = dlf.nn.ReLU()
dropout = dlf.nn.Dropout(0.2)

# Create a sequence of layers
model = dlf.nn.Sequential(
    dlf.nn.Linear(784, 256),
    dlf.nn.ReLU(),
    dlf.nn.Dropout(0.2),
    dlf.nn.Linear(256, 10)
)
```

### dlf.optim

The optimizer module provides optimizers for training neural networks.

```python
import dlf

# Create an optimizer
optimizer = dlf.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = dlf.optim.Adam(model.parameters(), lr=0.001)

# Training
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### dlf.data

The data module provides utilities for data loading and preprocessing.

```python
import dlf

# Create a dataset
dataset = dlf.data.Dataset(X, y)
dataloader = dlf.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over batches
for inputs, targets in dataloader:
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
```

## Logging System

### Logger Class

The `Logger` class provides a singleton-based logging system using spdlog.

```cpp
namespace dlf {

class Logger {
public:
    // Get the singleton instance
    static Logger& getInstance();

    // Initialize the logger with a log file
    void initialize(const std::string& log_file = "dlf.log");

    // Flush the log buffer to disk
    void flush();

    // Get the underlying spdlog logger
    [[nodiscard]] std::shared_ptr<spdlog::logger> getLogger() const;
};

} // namespace dlf
```

### Logging Macros

The framework provides convenience macros for logging at different levels:

```cpp
// Log at TRACE level
DLF_LOG_TRACE(...)

// Log at DEBUG level
DLF_LOG_DEBUG(...)

// Log at INFO level
DLF_LOG_INFO(...)

// Log at WARN level
DLF_LOG_WARN(...)

// Log at ERROR level
DLF_LOG_ERROR(...)

// Log at CRITICAL level
DLF_LOG_CRITICAL(...)
```

Each macro supports:
- String formatting using `{}` placeholders
- Multiple arguments
- Automatic source location tracking

### Example Usage

```cpp
#include <utils/logger.hpp>

// Initialize the logger
dlf::Logger::getInstance().initialize("my_log_file.log");

// Log a simple message
DLF_LOG_INFO("Starting application");

// Log with formatting
int value = 42;
DLF_LOG_INFO("The answer is {}", value);

// Log with multiple arguments
DLF_LOG_DEBUG("Tensor shape: {}x{}", rows, cols);

// Log an error
DLF_LOG_ERROR("Failed to load model: {}", error_message);

// Ensure logs are written to disk
dlf::Logger::getInstance().flush();
```

### Log Format

The default log format is:
```
[timestamp] [level] [file:line] message
```

Where:
- `timestamp`: ISO 8601 format with milliseconds
- `level`: Log level (trace, debug, info, warn, error, critical)
- `file:line`: Source file and line number
- `message`: The actual log message

Example:
```
[2024-04-08 09:33:30.220] [info] [main.cc:42] Application started
```

### Configuration

The logger can be configured with different patterns for console and file output:

```cpp
// Console sink (colorized)
console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");

// File sink
file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");
```

Pattern placeholders:
- `%Y-%m-%d %H:%M:%S.%e`: Timestamp
- `%l`: Log level
- `%s:%#`: Source file and line number
- `%v`: Message
- `%^` and `%$`: Color range (console only)

## Tensor Operations

[Back to Home](index.html) 