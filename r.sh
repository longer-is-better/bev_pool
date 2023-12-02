#!/bin/bash

function device_query() {
  mkdir -p build
  cat <<! > build/devicequery.cc
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("sm_%d%d\n", deviceProp.major, deviceProp.minor);
    }

    return 0;
}
!
  nvcc -O2 -o build/devicequery build/devicequery.cc
  ./build/devicequery
}
CC=$(device_query | head -n1)
CC_NUM=${CC##*_}

echo "Compute Capability: " $CC

function build_hpc() {
  mkdir -p hpc/build
  cd hpc/build
  if [ ! -f Makefile ]; then
    cmake  -DCMAKE_CUDA_ARCHITECTURES="$CC_NUM" ..
  fi
  make -j
  cd -
}

build_hpc
if [ ! -L data ]; then
  ln -s data.2 data
fi
nvcc -O3 -arch=$CC --use_fast_math bev_pool.cu -o bev_pool.so -shared -Xcompiler -fPIC

CUDA_LAUNCH_BLOCKING=1 python3 ./bev_pool.py
