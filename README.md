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
