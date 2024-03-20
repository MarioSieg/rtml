# RTML: RealTime Machine Learning
## Overview
RTML is a Machine Learning framework designed for both training and inference tasks on any GPU or CPU without vendor specific code such as CUDA.<br>
It leverages the power of Vulkan Compute to enable GPU acceleration across any GPU supported by Vulkan.<br>
Additionally, RTML provides a CPU backend optimized for SIMD operations and multi-threading for devices without a suitable GPU.<br>
With user-friendly Python bindings, RTML simplifies the integration of high-performance machine learning models into a variety of applications.<br>

## Features
* Cross-Platform GPU Support: Utilize any GPU that supports Vulkan for ML computations without needing specific GPU code.
* CPU Backend: Optimized for SIMD and multi-threading for systems without a proper GPU.
* Asynchronous Queues: Leverage Vulkan compute with asynchronous queues for efficient task management.
* Python Bindings: Easy-to-use Python interface for seamless integration into your projects.
* Real-Time Performance: Designed for real-time applications requiring rapid training and inference.

## Tensors in RTML
* Dimensions: Support for 1-4 dimensional tensors.
* Computation Graph: Tensors act as nodes within a Directed Acyclic Graph (DAG).
* JIT Compilation: Graphs can be Just-In-Time compiled for CPU or Vulkan backends.
* Execution: Supports Vulkan synchronization and multithreading.

## Workflow
Generate Computation Graph -> Validate -> Optimize -> Compile -> Execute

