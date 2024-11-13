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
void ContiguousKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  show_kernel("Contiguous type=" << dnn_support::type2String<T>::name()<<" x dims="<<x.dims()<<"out dims="<<out->dims());
  auto x_dims = x.dims();
  //auto out_dims = ValidateShape(shape.GetData(), x_dims);
  auto out_dims = x_dims;
  out->Resize(out_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  if (x.dims()[0] == out->dims()[0]) {
    out->share_lod(x);
  }

  if (!(x.initialized() && x.Holder() == out->Holder())) {
    show_debug(
        "Reshape type initialized=" << dnn_support::type2String<T>::name());
    dev_ctx.Alloc(out, x.dtype());
    auto dims = out->dims();
    auto x_data = x.data<T>();
    auto out_data = out->data<T>();

    void* stream = const_cast<void*>(dev_ctx.stream());
    auto* q = static_cast<sycl::queue*>(stream);
    q->memcpy(out_data, x_data, x.numel() * sizeof(T));

    out->Resize(dims);
    out->ResetLoD(x.lod());
  }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(contiguous,
                    SYCL,
                    ALL_LAYOUT,
                    custom_kernel::ContiguousKernel,
                    float,
                    double,
                    int8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    uint8_t,
                    bool,
                    phi::dtype::float16) {}
