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

#include <vector>

#include "kernels/dnn_support.hpp"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T, typename VType>
void FullValue(const phi::Context& dev_ctx,
               phi::DenseTensor* tensor,
               VType val) {
  show_kernel("FullValue type=" << dnn_support::type2String<T>::name());
  T* out_data = dev_ctx.Alloc<T>(tensor);

  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());
  auto num = tensor->numel();
  show_debug("FullValue size=" << num << " sizeof(T)=" << sizeof(T));
  std::cout<<"=========== call Full value Before =============="<<std::endl;
  //auto e = q->submit([&](sycl::handler& h) { h.fill(out_data, val, num); });
  
  VType* data = new VType[num];
  for(int i=0; i<num; i++) data[i] = val;
  auto data_ptr = static_cast<const VType*>(data);
  q->memcpy(out_data, data_ptr, tensor->memory_size());
  q->wait();
  std::cout<<"=========== call Full value finish =============="<<std::endl;
}

template <typename T>
void FullKernel(const phi::Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  auto int_shape = shape.GetData();
  out->Resize(std::vector<int64_t>(int_shape.cbegin(), int_shape.cend()));
  FullValue<T>(dev_ctx, out, val.to<T>());
}
}  // namespace custom_kernel


PD_BUILD_PHI_KERNEL(full,
                    SYCL,
                    ALL_LAYOUT,
                    custom_kernel::FullKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}