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

#include "kernels/dnn_support.hpp"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void EmbeddingSYCL(const phi::Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& weight,
                      int64_t padding_idx,
                      phi::DenseTensor* out){

    show_kernel("Embedding type=" << dnn_support::type2String<T>::name() << "size:"<<out->numel());
    
    int num = out->numel();
    std::vector<int> fake_data(2, num);

    dev_ctx.template Alloc<T>(out);
    auto* output = out->data<T>();

    auto* q = static_cast<sycl::queue*>(dev_ctx.stream());
    q->memcpy(output, &fake_data[0], num * sizeof(T));
    q->wait();
    
}

template <typename T>
void EmbeddingKernel(const phi::Context& ctx,
                     const phi::DenseTensor& input,
                     const phi::DenseTensor& weight,
                     int64_t padding_idx,
                     phi::DenseTensor* out) {
  EmbeddingSYCL<T>(ctx, input, weight, padding_idx, out);

}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(embedding,
                    SYCL,
                    ALL_LAYOUT,
                    custom_kernel::EmbeddingKernel,
                    float,
                    double,
                    int32_t,
                    int64_t,
                    int8_t,
                    phi::dtype::float16) {}