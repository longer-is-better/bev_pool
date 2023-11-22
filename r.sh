nvcc -O2 -arch=sm_87 --use_fast_math bev_pool.cu -o bev_pool.so -shared -Xcompiler -fPIC
mkdir -p hpc/b; cd hpc/b && make -j && cd -

CUDA_LAUNCH_BLOCKING=1 python3 ./bev_pool.py
