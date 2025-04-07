---
layout: default
title: Examples
---

# DLF Examples

## Computer Vision

### Image Classification with MNIST

```python
import dlf
import dlf.data.datasets as datasets

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
test_dataset = datasets.MNIST(root='./data', train=False)

# Create data loaders
train_loader = dlf.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = dlf.data.DataLoader(test_dataset, batch_size=1000)

# Define a CNN model
class CNN(dlf.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = dlf.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = dlf.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = dlf.nn.Dropout2d(0.25)
        self.dropout2 = dlf.nn.Dropout2d(0.5)
        self.fc1 = dlf.nn.Linear(9216, 128)
        self.fc2 = dlf.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = dlf.nn.functional.relu(x)
        x = self.conv2(x)
        x = dlf.nn.functional.relu(x)
        x = dlf.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = dlf.flatten(x, 1)
        x = self.fc1(x)
        x = dlf.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Create model, loss function, and optimizer
model = CNN().to("cuda")
loss_fn = dlf.nn.CrossEntropyLoss()
optimizer = dlf.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
```

## Natural Language Processing

### Text Classification

```python
import dlf
import dlf.data.datasets as datasets

# Define a simple RNN model
class RNN(dlf.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = dlf.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = dlf.nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = dlf.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Create model, loss function, and optimizer
model = RNN(vocab_size=10000, embedding_dim=100, hidden_dim=256, output_dim=2)
loss_fn = dlf.nn.CrossEntropyLoss()
optimizer = dlf.optim.Adam(model.parameters())

# Training loop example
for epoch in range(5):
    for batch in train_loader:
        text, labels = batch
        
        optimizer.zero_grad()
        predictions = model(text)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
```

[Back to Home](index.html) 