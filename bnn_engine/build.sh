#!/bin/bash
rm -rf build/*
cd build
cmake -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_PREFIX_PATH="/home/bodo/.pyenv/versions/xpu/lib/python3.10/site-packages/torch/share/cmake/Torch" \
      -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
      -DPYTHON_EXECUTABLE=$(which python) \
      -DPYBIND11_FINDPYTHON=ON \
      ..
make -j$(nproc)
