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

[Back to Home](index.html) 