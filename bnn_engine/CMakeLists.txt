cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(BNNEngine CXX)

# Require C++17 for SYCL 2020 and PyTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# 1. Compiler Flags (The Speed Knobs)
# ----------------------------------------------------------------------------
# -fsycl:           Enables Intel DPC++ SYCL compilation
# -O3:              Maximum optimization
# -march=native:    Tells the compiler to use AVX-512 if your CPU supports it
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -O3 -march=native -ffast-math")

# ----------------------------------------------------------------------------
# 2. Find Dependencies (PyTorch / LibTorch)
# ----------------------------------------------------------------------------
# You must pass -DCMAKE_PREFIX_PATH=/path/to/libtorch when running cmake
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIB torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
find_package(pybind11 REQUIRED)

# Include our headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----------------------------------------------------------------------------
# 3. Target A: The PyTorch/Python Binding Module
# ----------------------------------------------------------------------------
# pybind11_add_module automatically handles the Python .so extension mapping
pybind11_add_module(bnn_pytorch_ext
    bindings/python/bind_pytorch.cpp
)
# Link PyTorch ATen libraries and SYCL runtime
target_link_libraries(bnn_pytorch_ext PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB} sycl)

# ----------------------------------------------------------------------------
# 4. Target B: The Generic Shared Library (For Julia)
# ----------------------------------------------------------------------------
add_library(bnn_engine SHARED
    bindings/julia/bind_julia.cpp
)
target_link_libraries(bnn_engine PUBLIC sycl)
