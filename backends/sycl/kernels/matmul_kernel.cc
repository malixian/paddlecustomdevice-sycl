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

    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    auto* q = static_cast<sycl::queue*>(const_cast<void*>(ctx.stream()));
    auto eng =
        dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
    auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

    show_kernel("MatmulOneDNN()" << " type=" << dnn_support::type2String<T>::name() << " x dims="<<x.dims()<< " y dims="<<y.dims()<<" out dims="<<out->dims());

    // Source (src), weights and destination (dst) tensors dimensions.
    auto x_data = x.data<T>();
    auto y_data = y.data<T>();
    auto out_data = ctx.template Alloc<T>(out);

    auto x_dims = x.dims();
    auto y_dims = y.dims();
    auto out_dims = out->dims();

    phi::update_broadcast(y_dims, out_dims, -1);

    std::cout<<"broadcast y_dims: "<<y_dims<<std::endl;

    
    if (x_dims.size() > 3 && x_dims[0] == 1) {
        std::vector<int64_t> tmp(3, 1);
        for(int i=0; i<3; i++) tmp[i] = x_dims[i+1];
        x_dims = tmp;
    }
    if (y_dims.size() > 3 && y_dims[0] == 1) {
        std::vector<int64_t> tmp(3, 1);
        for(int i=0; i<3; i++) tmp[i] = y_dims[i+1];
        y_dims = tmp;
    }
    if (out_dims.size() > 3 && out_dims[0] == 1) {
        std::vector<int64_t> tmp(3, 1);
        for(int i=0; i<3; i++) tmp[i] = out_dims[i+1];
        out_dims = tmp;
    }

    std::cout<<"reduced dims: "<<x_dims<<" "<<y_dims<<" "<<out_dims<<std::endl;

    auto src_md = dnnl::memory::desc(x_dims, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(x_dims));
    auto weights_md = dnnl::memory::desc(y_dims, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(y_dims));
    auto dst_md = dnnl::memory::desc(out_dims, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(out_dims));

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
    std::cout<<"mat mul kernel end..........."<<std::endl; 
}

}  // namespace custom_kernel


PD_BUILD_PHI_KERNEL(matmul,
                    SYCL,
                    ALL_LAYOUT,
                    custom_kernel::MatmulKernel,
                    float,
                    phi::dtype::float16) {}
