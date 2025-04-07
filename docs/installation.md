---
layout: default
title: Installation
---

# Installation

## Installing via pip

The simplest way to install DLF is via pip:

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

[Back to Home](index.html) 