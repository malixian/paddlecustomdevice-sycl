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
#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void AssignValueKernel(const phi::Context& dev_ctx,
                       const std::vector<int>& shape,
                       phi::DataType dtype,
                       const std::vector<phi::Scalar>& values,
                       phi::DenseTensor* out) {
  show_kernel("AssignValue-SYCL, type=" << dnn_support::type2String<T>::name());

  auto template_dtype = phi::capi::CppTypeToPDType<T>::Type();
  PD_CHECK(dtype == template_dtype,
           "Argument dtype mismatch for kernel dtype, "
           "argument dtype is %s, kernel dtype is %s.",
           dtype,
           template_dtype);
  auto out_size = values.size();
  out->Resize({static_cast<int64_t>(out_size)});
  auto out_data = dev_ctx.template Alloc<T>(out);

  std::vector<T> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
    assign_values.emplace_back(val.to<T>());
  }
  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());
  q->memcpy(out_data, &assign_values[0], assign_values.size() * sizeof(T));
  q->wait();
  out->Resize(std::vector<int64_t>(shape.cbegin(), shape.cend()));
}

template <typename T>
void AssignKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto x_data = x.data<T>();
  std::memcpy(out_data, x_data, sizeof(T) * x.numel());
}

template <typename T>
void AssignRawKernel(const phi::Context& dev_ctx,
                     const paddle::optional<phi::DenseTensor>& x,
                     phi::DenseTensor* out) {
  show_kernel("AssignRaw-SYCL, type=" << dnn_support::type2String<T>::name());

  if (x) {
    if (!x->initialized()) {
      return;
    }
    auto x_data = x->data<T>();
    auto out_data = dev_ctx.template Alloc<T>(out);

    auto* q = static_cast<sycl::queue*>(dev_ctx.stream());
    q->memcpy(out_data, x_data, x->numel());
    q->wait();
  }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(assign_value,
                    SYCL,
                    ALL_LAYOUT,
                    custom_kernel::AssignValueKernel,
                    int,
                    int64_t,
                    float,
                    double,
                    phi::dtype::float16) {}

PD_BUILD_PHI_KERNEL(assign_raw,
                    SYCL,
                    ALL_LAYOUT,
                    custom_kernel::AssignRawKernel,
                    int,
                    int64_t,
                    float,
                    double,
                    phi::dtype::float16) {}
