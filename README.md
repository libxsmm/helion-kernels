# Helion Kernels
A collection of Helion kernels and their equivalent PyTorch models with example inputs to measure their performance.

## Dependencies
* [Helion](https://github.com/pytorch/helion)
* [optional] [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton)
  * XPU requires special [nightly wheels](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml) of PyTorch and Triton

## Kernels
The available kernels are based on and follow [KernelBench](https://github.com/ScalingIntelligence/KernelBench) categories:
* Level 1: Single-kernel operators - The foundational building blocks of neural nets
* Level 2: Simple fusion patterns - A fused kernel would be faster than separated kernels
* Level 3: Full model architectures - Optimize entire model architectures end-to-end
