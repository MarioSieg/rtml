# RTML: Real-Time Machine Learning Framework

## Overview

RTML is a versatile machine learning framework designed to integrate seamlessly into real-time applications such as video games, computer vision systems, autonomous vehicles, and robotics. Its primary goal is to enable dynamic, efficient, and adaptable machine learning inference within environments where computation time and resources are critically constrained.

## Key Features

- **Dynamic Inference with Replay Trees:** At the heart of RTML are the Replay Trees, which are Directed Acyclic Graph (DAG)-based computation graphs, which can be baked and JIT-compiled. These allow for flexible time slicing and dynamic morphing to accommodate the varying computation budgets of each frame, ensuring consistent performance without compromising on decision-making quality.

- **Real-Time Optimization:** RTML is engineered for real-time scenarios. It can dynamically adjust its computational load to fit within the per-frame computation budget, ensuring that machine learning tasks do not interfere with the core simulation loop of your application.

- **Multi-Platform Support:** RTML is designed to operate across various computing environments. It supports both CPUs and GPUs, catering to a wide range of application needs and hardware specifications.

- **Asynchronous Processing:** The framework supports fully asynchronous operation, allowing machine learning tasks to run in parallel without disrupting the primary simulation or application loop. This feature is critical for maintaining smooth and responsive real-time applications.

- **Integrated Learning and Inference:** RTML supports coalesced training and inference, enabling real-time learning and adaptation. This feature is particularly useful in applications like robotics and autonomous systems, where on-the-fly learning and adjustment are crucial.

## Use Cases

RTML is particularly well-suited for applications where real-time performance and adaptive machine learning capabilities are paramount. These include:

- **Video Games:** Enhance gaming experiences with adaptive AI that learns and evolves in real-time, creating more engaging and challenging environments.

- **Computer Vision:** Implement real-time object detection, tracking, and analysis for applications such as surveillance, traffic monitoring, or interactive installations.

- **Autonomous Vehicles:** Empower self-driving cars with the ability to make split-second decisions and adjustments based on real-time environmental data.

- **Robotics:** Enable robots to learn from their environment and adapt to new tasks and challenges on the fly.

By leveraging RTML, developers can integrate advanced machine learning capabilities into their real-time applications without sacrificing performance or responsiveness. Whether you're building the next generation of interactive entertainment or developing critical autonomous systems, RTML provides the tools and flexibility needed to bring your vision to life.


## Getting Started
RTML runtime is written in C++ 20 and provides:
* A Python API for easy integration with Python-based applications.
* A Lua JIT FFI API for easy integration with Lua-based applications.