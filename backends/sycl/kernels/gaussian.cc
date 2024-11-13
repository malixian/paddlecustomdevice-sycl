// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <random>

#include "kernels/dnn_support.hpp"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void GaussianKernel(const phi::Context &ctx,
                    const phi::IntArray &shape,
                    float mean,
                    float std,
                    int seed,
                    phi::DataType dtype,
                    phi::DenseTensor* out)
{

  show_kernel("Call Gaussian Kernel shape = "<<shape.GetData());
  auto shape_data = shape.GetData();
  out->Resize(std::vector<int64_t>(shape_data.begin(), shape_data.end()));
  auto out_data = ctx.template Alloc<T>(out);
  auto numel = out->numel();

  phi::DenseTensor cpu_tensor;
  cpu_tensor.Resize(std::vector<int64_t>(shape_data.begin(), shape_data.end()));
  cpu_tensor.set_dtype(out->dtype());
  T* cpu_data = ctx.template HostAlloc<T>(&cpu_tensor);
  std::normal_distribution<float> dist(mean, std);

  std::shared_ptr<std::mt19937_64> engine;
  engine = std::make_shared<std::mt19937_64>();
  if (seed) {
    engine->seed(seed);
  } else {
    std::random_device device;
    auto rseed = (static_cast<uint64_t>(device()) << 32) | device();
    engine->seed(rseed);
  }

  for (int64_t i = 0; i < numel; ++i) {
    cpu_data[i] = static_cast<T>(dist(*engine));
    //std::cout<<cpu_data[i]<<" ";
  }

  auto* q = static_cast<sycl::queue*>(ctx.stream());
  q->memcpy(out_data, cpu_data, numel * sizeof(T));
  q->wait();

}

}  // namespace custom_kernel


PD_BUILD_PHI_KERNEL(gaussian,
                    SYCL,
                    ALL_LAYOUT,
                    custom_kernel::GaussianKernel,
                    float,
                    double,
                    phi::dtype::float16) {}
