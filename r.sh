nvcc -O3 -arch=sm_80 --use_fast_math bev_pool.cu -o bev_pool.so -shared -Xcompiler -fPIC

CUDA_LAUNCH_BLOCKING=1 python3 bev_pool.py
