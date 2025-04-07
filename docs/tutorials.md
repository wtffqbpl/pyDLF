---
layout: default
title: Tutorials
---

# DLF Tutorials

## Basic Tutorials

### Getting Started with Tensors

Learn the basics of creating and manipulating tensors in DLF:

```python
import dlf

# Create a tensor
x = dlf.tensor([1, 2, 3, 4])
y = dlf.tensor([[1, 2], [3, 4]])

# Basic operations
z = x + y
w = x * 2
```

### Training a Simple Neural Network

```python
import dlf

# Define a model
model = dlf.nn.Sequential(
    dlf.nn.Linear(784, 128),
    dlf.nn.ReLU(),
    dlf.nn.Linear(128, 10)
)

# Define loss function and optimizer
loss_fn = dlf.nn.CrossEntropyLoss()
optimizer = dlf.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

## Advanced Tutorials

### Custom Neural Network Layers

Learn how to create custom neural network layers:

```python
import dlf

class CustomLayer(dlf.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = dlf.Parameter(dlf.randn(in_features, out_features))
        self.bias = dlf.Parameter(dlf.zeros(out_features))
        
    def forward(self, x):
        return dlf.matmul(x, self.weight) + self.bias
```

### Distributed Training

Learn how to train models across multiple GPUs:

```python
import dlf

# Initialize distributed environment
dlf.distributed.init_process_group(backend='nccl')

# Create a distributed model
model = dlf.nn.parallel.DistributedDataParallel(model)

# Create a distributed dataloader
sampler = dlf.data.distributed.DistributedSampler(dataset)
dataloader = dlf.data.DataLoader(dataset, sampler=sampler)
```

[Back to Home](index.html) 