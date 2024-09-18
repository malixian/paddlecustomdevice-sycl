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
void FullKernel(const phi::Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  
  
  show_kernel(
      "Full-ONEDNN type=" << dnn_support::type2String<T>::name()<<" shape= "<<shape.GetData()<<" out dims:"<<out->dims()<<" full val:"<<val.to<T>());

  auto int_shape = shape.GetData();
  out->Resize(std::vector<int64_t>(int_shape.cbegin(), int_shape.cend()));
  auto out_data = dev_ctx.template Alloc<T>(out);

  T fill_value = val.to<T>();

  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));

  auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

  dnnl::memory::dims io_dims = out->dims();
  auto src_md = dnnl::memory::desc(io_dims, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(io_dims));
  auto dst_md = dnnl::memory::desc(io_dims, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(io_dims));

  auto src_mem = dnnl::memory(src_md, eng, out_data);
  auto dst_mem = dnnl::memory(dst_md, eng, out_data);

  auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eng,
            dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_linear, src_md,
            dst_md, 0.f, fill_value);
  auto eltwise_prim = dnnl::eltwise_forward(eltwise_pd);

  // Primitive arguments.
  std::unordered_map<int, dnnl::memory> eltwise_args;
  eltwise_args.insert({DNNL_ARG_SRC, src_mem});
  eltwise_args.insert({DNNL_ARG_DST, dst_mem});

  eltwise_prim.execute(engine_stream, eltwise_args);
  engine_stream.wait();
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
                    bool
                    ) {}
