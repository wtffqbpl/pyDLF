# DLF (Deep Learning Framework)

A GPU-accelerated deep learning framework built for high-performance AI research and applications.

## Overview

DLF is a flexible and efficient library for deep learning that provides GPU acceleration similar to PyTorch. It aims to provide researchers and developers with a powerful toolset for building and training neural networks.

## Features

- GPU-accelerated tensor operations
- Automatic differentiation engine
- Neural network building blocks
- Optimizers for model training
- Data loading and preprocessing utilities
- Support for distributed training
- Comprehensive logging system with file and console output
- Unit testing with Google Test

## Installation

```bash
pip install pyDLF
```

## Build from Source

If you prefer to build DLF from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pyDLF.git
   cd pyDLF
   ```

2. Set up the build environment:

   For macOS (Workaround):
   ```bash
   export CPLUS_INCLUDE_PATH="$(xcrun --show-sdk-path)/usr/include/c++/v1:${CPLUS_INCLUDE_PATH}"
   ```

   For Linux:
   ```bash
   # Install required dependencies for your distribution
   # e.g., for Ubuntu:
   sudo apt-get install build-essential cmake
   ```

3. Build and install:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   pip install -e .
   ```

## Requirements

- Python 3.7+
- CUDA Toolkit 11.0+ (for GPU acceleration)
- NumPy

## Quick Start

```python
import dlf

# Create tensors
x = dlf.tensor([1, 2, 3, 4], device="cuda")
y = dlf.tensor([5, 6, 7, 8], device="cuda")

# Operations
z = x + y
print(z)  # tensor([6, 8, 10, 12], device="cuda")

# Create a simple neural network
model = dlf.nn.Sequential(
    dlf.nn.Linear(784, 128),
    dlf.nn.ReLU(),
    dlf.nn.Linear(128, 10)
)

# Define loss function and optimizer
loss_fn = dlf.nn.CrossEntropyLoss()
optimizer = dlf.optim.SGD(model.parameters(), lr=0.01)

# Training loop example
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Documentation

For full documentation, visit [pyDLF](https://wtffqbpl.github.io/pyDLF/).

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md) to get started.

## License

DLF is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Logging System

The framework includes a comprehensive logging system based on spdlog. The logger provides:

- Multiple log levels (TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL)
- Both file and console output
- Source file and line number tracking
- Colorized console output
- Automatic log file rotation
- Thread-safe logging

### Usage

```cpp
#include <utils/logger.hpp>

// Initialize the logger
dlf::Logger::getInstance().initialize("my_log_file.log");

// Log messages at different levels
DLF_LOG_TRACE("Trace message");
DLF_LOG_DEBUG("Debug message");
DLF_LOG_INFO("Info message");
DLF_LOG_WARN("Warning message");
DLF_LOG_ERROR("Error message");
DLF_LOG_CRITICAL("Critical message");

// Log with formatting
int value = 42;
DLF_LOG_INFO("The answer is {}", value);

// Ensure logs are written to disk
dlf::Logger::getInstance().flush();
```

### Log Format

The default log format includes:
- Timestamp: `[%Y-%m-%d %H:%M:%S.%e]`
- Log level: `[%l]`
- Source location: `[%s:%#]`
- Message: `%v`

Example log output:
```
[2024-04-08 09:33:30.220] [info] [logger_test.cc:111] Test pattern
```

## Building

### Prerequisites

- C++20 compatible compiler
- CMake 3.20 or higher
- Python 3.8 or higher
- spdlog
- pybind11
- Google Test

### Build Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/pydlf.git
cd pydlf

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
cmake --build .

# Run tests
ctest
```

## Testing

The framework includes comprehensive unit tests using Google Test. To run the tests:

```bash
cd build
ctest --output-on-failure
```

## Documentation

Detailed documentation is available in the `docs` directory:

- [API Reference](docs/api.md)
- [Tutorial](docs/tutorial.md)
- [Contributing Guide](docs/contributing.md)
