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
#include <deque>

namespace custom_kernel {


template <typename T>
void CompareKernelDNN(const phi::Context& dev_ctx,
                         std::string kernel_name,
                         dnnl::algorithm binary_type,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         int axis,
                         phi::DenseTensor* out) {
  show_kernel(kernel_name << "-DNN type="
                          << dnn_support::type2String<T>::name()<<" x_dims:"<<x.dims()<<" y_dims:"<<y.dims()<<" out_dims:"<<out->dims()<<" axis:"<<axis);

  void* stream = const_cast<void*>(dev_ctx.stream());
  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));

  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

  auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

  dnnl::memory::dims dims_x = x.dims();
  dnnl::memory::dims dims_y = y.dims();
  dnnl::memory::dims dims_out = out->dims();

  phi::update_broadcast(dims_x, dims_y, axis);

  auto md_x = dnnl::memory::desc(
      dims_x, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(dims_x));

  auto md_y = dnnl::memory::desc(
      dims_y, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(dims_y));
  auto md_out = dnnl::memory::desc(dims_out,
                                   dnn_support::toDnnType<T>::type,
                                   dnn_support::dims2Tag(dims_out));

  auto x_mem = dnnl::memory(md_x, eng, x.data<T>());
  auto y_mem = dnnl::memory(md_y, eng, y.data<T>());

  auto out_data = dev_ctx.template Alloc<T>(out);

  auto out_mem = dnnl::memory(md_out, eng, out_data);

  //auto oper_desc = dnnl::binary::desc(binary_type, md_x, md_y, md_out);
  auto prim_desc = dnnl::binary::primitive_desc(eng, binary_type, md_x, md_y, md_out);
  auto prim = dnnl::binary(prim_desc);

  std::unordered_map<int, dnnl::memory> binary_args;
  binary_args.insert({DNNL_ARG_SRC_0, x_mem});
  binary_args.insert({DNNL_ARG_SRC_1, y_mem});
  binary_args.insert({DNNL_ARG_DST, out_mem});

  prim.execute(engine_stream, binary_args);
  engine_stream.wait();
} 

template <typename T>
void NotEqualKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  std::cout<<"call not equal kernel"<<std::endl;
  CompareKernelDNN<T>(
      dev_ctx,
      "NotEqual",
      dnnl::algorithm::binary_ne,
      x,
      y,
      axis,
      out
      );
}

template <typename T>
void EqualKernel(const phi::Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 int axis,
                 phi::DenseTensor* out) {
  CompareKernelDNN<T>(
      dev_ctx,
      "Equal",
      dnnl::algorithm::binary_eq,
      x,
      y,
      axis,
      out);
}

template <typename T>
void LessThanKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  CompareKernelDNN<T>(dev_ctx,
                   "LessThanKernel",
                   dnnl::algorithm::binary_lt,
                   x,
                   y,
                   axis,
                   out
                   );
}

template <typename T>
void LessEqualKernel(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  CompareKernelDNN<T>(dev_ctx,
                   "LessEqual",
                   dnnl::algorithm::binary_le,
                   x,
                   y,
                   axis,
                   out
                   );
}

template <typename T>
void GreaterThanKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  CompareKernelDNN<T>(dev_ctx,
                   "GreaterThan",
                   dnnl::algorithm::binary_gt,
                   x,
                   y,
                   axis,
                   out
                   );
}

template <typename T>
void GreaterEqualKernel(const phi::Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        int axis,
                        phi::DenseTensor* out) {
  CompareKernelDNN<T>(dev_ctx,
                   "GreaterEqual",
                   dnnl::algorithm::binary_ge,
                   x,
                   y,
                   axis,
                   out);
}

}  // namespace custom_kernel

#define PD_REGISTER_COMPARE_KERNEL(name, func)            \
  PD_BUILD_PHI_KERNEL(name,                               \
                      SYCL,                               \
                      ALL_LAYOUT,                         \
                      custom_kernel::func##Kernel,        \
                      float,                              \
                      double,                             \
                      uint8_t,                            \
                      int16_t,                            \
                      int32_t,                            \
                      int64_t                           ) {}
PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(greater_equal, GreaterEqual)
PD_REGISTER_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPARE_KERNEL(not_equal, NotEqual)
