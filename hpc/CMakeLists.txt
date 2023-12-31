cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(bev_pool LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda packages
find_package(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
  set(CUDA_ARCHITECTURE_FLAGS "87")
endif()

# set nvcc's options
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                    "-U__CUDA_NO_HALF_OPERATORS__"
                    "-U__CUDA_NO_HALF_CONVERSIONS__"
                    "-U__CUDA_NO_HALF2_OPERATORS__"
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"
                    "--expt-relaxed-constexpr"
                    "--expt-extended-lambda"
                    "--use_fast_math")
                    # "--gencode arch=compute_86,code=sm_86")

message(STATUS "CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}")

# set include file
include_directories("/usr/local/cuda/include") # cuda include dir
include_directories("./") # current include dir
# set lib file
link_directories("/usr/local/cuda/lib64")
# set lib source file
file(GLOB_RECURSE LIB_SRC "./*.cu")
message(STATUS "LIB_SRC = ${LIB_SRC}")

# build shared lib
add_library(bev_pool_shared SHARED ${LIB_SRC})
# build executed file
add_executable(
    bev_pool
    "main.cpp"
    "cuda_gadget.cpp"
    "cnpy.cpp"
    )
# set link lib for executed file
target_link_libraries(
    bev_pool
    bev_pool_shared
    -lcuda
    -lcudart
    -lz
    )

