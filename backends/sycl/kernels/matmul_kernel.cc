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

#include <iostream>

namespace custom_kernel {
template <typename T>
void MatmulKernel(const phi::Context& ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out) {

    std::cout<<"=========== call oneDNN Matmul Before =============="<<std::endl;

    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    auto* q = static_cast<sycl::queue*>(const_cast<void*>(ctx.stream()));
    auto eng =
        dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
    auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

    // Source (src), weights and destination (dst) tensors dimensions.
    auto x_data = x.data<T>();
    auto y_data = y.data<T>();
    auto out_data = ctx.template Alloc<T>(out);

    auto src_md = dnnl::memory::desc(x.dims(), dnn_support::toDnnType<T>::type, tag::ab);
    auto weights_md = dnnl::memory::desc(y.dims(), dnn_support::toDnnType<T>::type, tag::ab);
    auto dst_md = dnnl::memory::desc(out->dims(), dnn_support::toDnnType<T>::type, tag::ab);

    auto src_mem = dnnl::memory(src_md, eng, x_data);
    auto weights_mem = dnnl::memory(weights_md, eng, y_data);
    auto dst_mem = dnnl::memory(dst_md, eng, out_data);

    // Create primitive descriptor.
    auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_md, weights_md, dst_md);

    // Create the primitive.
    auto matmul_prim = dnnl::matmul(matmul_pd);

    // Primitive arguments.
    std::unordered_map<int, dnnl::memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});

    // Execution.
    matmul_prim.execute(engine_stream, matmul_args);
    engine_stream.wait();

    std::cout<<"=========== call oneDNN Matmul Finish =============="<<std::endl;
 
}

}  // namespace custom_kernel


PD_BUILD_PHI_KERNEL(matmul,
                    SYCL,
                    ALL_LAYOUT,
                    custom_kernel::MatmulKernel,
                    float,
                    phi::dtype::float16) {}