# SeqConvNet-Framework

## Overview

`SeqConvNet-Framework` is the production-ready and standardized implementation of the **SeqConvNet** algorithm. This project leverages our self-developed **`pyvirid` message scheduling engine** as the control flow core, perfectly decoupling the data flow, control flow, and algorithmic logic.

Designed with a modular monorepo structure, the project is divided into three core sub-packages:

- **📦 seqconvnet-core**: Provides out-of-the-box, high-performance point cloud DataLoaders, foundational geometric data structures, cutting-edge hybrid network modules (CNN / RNN / Transformer), and high-precision evaluators via standard APIs.
- **🛠️ seqconvnet-preprocess**: A complete data preprocessing pipeline driven by `pyvirid`. Provided as an extension package, it supports large-scale point cloud voxelization and geometric slicing.
- **🔥 seqconvnet-train**: A comprehensive network training and lifecycle management workflow driven by `pyvirid`. Provided as an extension package, it gracefully implements multi-stage metric monitoring and self-driven training through an event-driven architecture.

## Architectural Features

At its core, `SeqConvNet` adopts a unique **"Sequence-Convolution-Sequence"** hybrid topology to perform semantic segmentation on 3D point clouds.

- **Ultra-Fast Inference**: Through optimized dimension transformations and feature reuse, it stands as one of the fastest known network topologies for point cloud semantic segmentation inference.
- **Event-Driven**: Thanks to the `pyvirid` engine, the entire framework lifecycle (from `Startup` and `Epoch` iterations to `Eval` assessment) is entirely driven by non-blocking messages (`Message` & `System`), offering exceptional scalability and determinism.

## Getting Started

Detailed scaffolding examples are provided in the `examples` directory to help you get started quickly:

- **Basic API Usage**: Please refer to the component instantiation examples under `examples/core/`.
- **One-Click Data Preprocessing**: Please refer to `examples/preprocess.py`.
- **Launch Event-Driven Training**: Please refer to `examples/train.py`.

```python
# Example: Run the training workflow
python examples/train.py
```

## Roadmap & TODOs

- [ ] 📥 **Robust Model Lifecycle Management**: Provide features for model checkpoint saving and seamless loading.
- [ ] 🤖 **Self-Supervised Learning (SSL) Support**: Introduce a self-supervised pre-training pipeline for point clouds to further leverage the potential of unlabeled data.
- [ ] 📚 **Full-Scenario Example Library**: Add end-to-end deployment demos for more real-world point cloud datasets (e.g., DALES / S3DIS).