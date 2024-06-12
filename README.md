# Keyword spotting on STM32 MCU
This repository provides a comprehensive framework for implementing keyword spotting on STM32 microcontrollers. 
The key components of this framework include:
* **PyTorch Framework**: Utilizes a filter bank to extract features from audio data and trains an RNN-GRU network.
* **TensorFlow Lite Framework**: Facilitates the training of a Depthwise Separable Convolutional Neural Network (DS-CNN). 
It supports Quantization-Aware Training (QAT) to optimize the network and generates a C header file for deployment on the MCU.
* **STM32 Project**: Contains the setup for deploying and running real-time keyword spotting on STM32 hardware. (Not public)
* **Demo Scripts**: Provided to facilitate running the demonstrations directly on the MCU.

For detailed instructions, please refer to the README.md files located in the respective directories within this repository.

