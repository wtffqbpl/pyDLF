---
layout: default
---

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

[Get Started with Installation](installation.html) | [View on GitHub](https://github.com/yourusername/pyDLF) 