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

template <typename T>
void FullValue(const phi::Context& dev_ctx,
               phi::DenseTensor* tensor,
               T val) {
  show_kernel("FullValue type=" << dnn_support::type2String<T>::name());
  auto num = tensor->numel();
  show_debug("FullValue size=" << num << " sizeof(T)=" << sizeof(T));
  //std::cout<<"=========== call Full value Before =============="<<std::endl;
  //auto e = q->submit([&](sycl::handler& h) { h.fill(out_data, val, num); });

  /* q->submit([&](sycl::handler& h) {
    h.parallel_for(num, [out_data, val](sycl::id<1> i) {
      out_data[i] = val;
    });
  }); */

  std::cout<< "out size: "<< num << " =============="<<std::endl;

  auto out_data = dev_ctx.template Alloc<T>(tensor);

  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());

  std::vector<T> assign_values;
  assign_values.reserve(num);
  for (int i=0; i<num; i++) {
    assign_values.emplace_back(val);
  }
  q->memcpy(out_data, &assign_values[0], assign_values.size() * sizeof(T));
  q->wait();

  std::cout<<"=========== call Full value::" << val<< "out size: "<< num << " =============="<<std::endl;
  //std::cout<<"=========== call Full value finish =============="<<std::endl;
}

template <typename T>
void FullKernel(const phi::Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  
  show_kernel(
      "FULL type=" << dnn_support::type2String<T>::name());
  auto int_shape = shape.GetData();
  auto tmp_vec = std::vector<int64_t>(int_shape.cbegin(), int_shape.cend());

  out->Resize(tmp_vec);
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto num = out->numel();
  num = 1;

  show_debug("FullValue size=" << num << " sizeof(T)=" << sizeof(T));
  std::cout<<"tmp vec: "<<tmp_vec<<std::endl;
  std::cout<<"=========== call Full value::" << val.to<T>()<< " out size: "<< num << " =============="<<std::endl;
  
  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());
  std::vector<T> assign_values;
  int assign_val = 1;
  if (int(val.to<T>()) >= 0 && int(val.to<T>()) <= 10000) assign_val = val.to<T>();
  std::cout<<"assign val: "<<assign_val<<std::endl;
  for (int i=0; i<num; i++) {
    assign_values.push_back(assign_val);
  }
  std::cout<<"cpu data ready"<<std::endl;
  q->memcpy(out_data, &assign_values[0], num * sizeof(T));
  q->wait();
  std::cout<<"full finish ready"<<std::endl;
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
                    int64_t
                    //bool,
                    //phi::dtype::float16
                    ) {}
