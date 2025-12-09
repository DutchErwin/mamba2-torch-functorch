#!/bin/bash



# CUDA 12.8 Blackwell Environment Setup

# For compiling CUDA extensions with sm_120 support

export CUDA_HOME=/home/minibeast/cuda-12.8-blackwell/cuda_nvcc

export PATH=$CUDA_HOME/bin:$PATH

# Library paths for linking

export LD_LIBRARY_PATH=/home/minibeast/cuda-12.8-blackwell/cuda_cudart/lib64:${LD_LIBRARY_PATH:-}

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

export LIBRARY_PATH=/home/minibeast/cuda-12.8-blackwell/cuda_cudart/lib64:${LIBRARY_PATH:-}

export LIBRARY_PATH=/home/minibeast/cuda-12.8-blackwell/cuda_cudart/lib64/stubs:$LIBRARY_PATH

export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH

# Base CUDA includes

export CPATH=/home/minibeast/cuda-12.8-blackwell/cuda_cudart/include:${CPATH:-}

export CPATH=/home/minibeast/cuda-12.8-blackwell/cuda_cccl/targets/x86_64-linux/include:$CPATH

# ALL PyTorch bundled CUDA headers

NVIDIA_PKG=/home/minibeast/miniforge/envs/hmsnet/lib/python3.12/site-packages/nvidia

export CPATH=$NVIDIA_PKG/cublas/include:$CPATH

export CPATH=$NVIDIA_PKG/cuda_cupti/include:$CPATH

export CPATH=$NVIDIA_PKG/cuda_nvrtc/include:$CPATH

export CPATH=$NVIDIA_PKG/cuda_runtime/include:$CPATH

export CPATH=$NVIDIA_PKG/cudnn/include:$CPATH

export CPATH=$NVIDIA_PKG/cufft/include:$CPATH

export CPATH=$NVIDIA_PKG/cufile/include:$CPATH

export CPATH=$NVIDIA_PKG/curand/include:$CPATH

export CPATH=$NVIDIA_PKG/cusolver/include:$CPATH

export CPATH=$NVIDIA_PKG/cusparse/include:$CPATH

export CPATH=$NVIDIA_PKG/cusparselt/include:$CPATH

export CPATH=$NVIDIA_PKG/nccl/include:$CPATH

export CPATH=$NVIDIA_PKG/nvjitlink/include:$CPATH

export CPATH=$NVIDIA_PKG/nvshmem/include:$CPATH

export CPATH=$NVIDIA_PKG/nvtx/include:$CPATH

export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "CUDA 12.8 Blackwell environment activated"

echo "  CUDA_HOME: $CUDA_HOME"

echo "  nvcc: $(which nvcc 2>/dev/null || echo 'not in PATH')"


