# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/mlx/repos/paddle-onednn-dev/paddlecustomdevice-onednn/Paddle/third_party/gflags"
  "/home/mlx/repos/paddle-onednn-dev/paddlecustomdevice-onednn/backends/sycl/build/third_party/gflags/src/extern_gflags-build"
  "/home/mlx/repos/paddle-onednn-dev/paddlecustomdevice-onednn/backends/sycl/build/third_party/gflags"
  "/home/mlx/repos/paddle-onednn-dev/paddlecustomdevice-onednn/backends/sycl/build/third_party/gflags/tmp"
  "/home/mlx/repos/paddle-onednn-dev/paddlecustomdevice-onednn/backends/sycl/build/third_party/gflags/src/extern_gflags-stamp"
  "/home/mlx/repos/paddle-onednn-dev/paddlecustomdevice-onednn/backends/sycl/build/third_party/gflags/src"
  "/home/mlx/repos/paddle-onednn-dev/paddlecustomdevice-onednn/backends/sycl/build/third_party/gflags/src/extern_gflags-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/mlx/repos/paddle-onednn-dev/paddlecustomdevice-onednn/backends/sycl/build/third_party/gflags/src/extern_gflags-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/mlx/repos/paddle-onednn-dev/paddlecustomdevice-onednn/backends/sycl/build/third_party/gflags/src/extern_gflags-stamp${cfgdir}") # cfgdir has leading slash
endif()
