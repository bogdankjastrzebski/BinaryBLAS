# BinaryBLAS
The binary blas implementation with thorin.

```bash
cmake -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_PREFIX_PATH="/home/bodo/.pyenv/versions/xpu/lib/python3.10/site-packages/torch/share/cmake/Torch" \
      -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
      ..
```

```bash
make -j$(nproc)
```


# Performance

[intel] ~/D/G/B/bnn_engine main ⨯  perf stat -e cycles,instructions
,L1-dcache-loads,L1-dcache-load-misses python benchmark_linear.py
Warming up... Matrix size: 1024x4096 -> 4096
Total time for 1000 iterations: 73.8121 seconds
Throughput: 13.55 inferences/sec

 Performance counter stats for 'python benchmark_linear.py':

 1,214,796,888,352      cpu_atom/cycles/                           
                             (55.93%)
 2,343,718,321,076      cpu_core/cycles/                           
                             (54.70%)
 1,265,218,053,657      cpu_atom/instructions/                     
                             (55.93%)
 2,891,460,682,619      cpu_core/instructions/                     
                             (54.70%)
   353,059,862,113      cpu_atom/L1-dcache-loads/                  
                             (55.93%)
   323,002,187,915      cpu_core/L1-dcache-loads/                  
                             (54.70%)
   <not supported>      cpu_atom/L1-dcache-load-misses/            
                           
       605,444,622      cpu_core/L1-dcache-load-misses/            
                             (54.70%)

      78.255687597 seconds time elapsed

    1320.221917000 seconds user
       2.293719000 seconds sys


[intel] ~/D/G/B/b/build main ⨯  cd ..                      01:45:20
                                perf stat -e cycles,instructions,L1
-dcache-loads,L1-dcache-load-misses python benchmark_linear.py
Warming up... Matrix size: 1024x4096 -> 4096
Total time for 1000 iterations: 74.3981 seconds
Throughput: 13.44 inferences/sec

 Performance counter stats for 'python benchmark_linear.py':

 1,209,505,223,809      cpu_atom/cycles/                           
                             (57.27%)
 2,146,153,254,855      cpu_core/cycles/                           
                             (55.55%)
 1,119,173,242,921      cpu_atom/instructions/                     
                             (57.27%)
 2,590,640,732,670      cpu_core/instructions/                     
                             (55.55%)
   240,703,413,024      cpu_atom/L1-dcache-loads/                  
                             (57.27%)
   198,647,301,058      cpu_core/L1-dcache-loads/                  
                             (55.55%)
   <not supported>      cpu_atom/L1-dcache-load-misses/            
                           
       539,473,716      cpu_core/L1-dcache-load-misses/            
                             (55.55%)

      78.813681952 seconds time elapsed

    1289.874299000 seconds user
       2.741578000 seconds sys


[intel] ~/D/G/B/bnn_engine main ⨯                          01:49:48

