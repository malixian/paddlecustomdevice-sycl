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
void Conv2dKernel(const phi::Context& ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  const std::string& padding_algorithm,
                  const std::vector<int>& dilations_t,
                  int groups,
                  const std::string& data_format,
                  phi::DenseTensor* output) {
    
    
    dnnl::memory::dims strides_dims = {strides_t[0], strides_t[1]};
    dnnl::memory::dims padding_dims_l = {paddings_t[0], paddings_t[1]};
    dnnl::memory::dims padding_dims_r = {paddings_t[0], paddings_t[1]};
    auto dilations = dilations_t;

    const bool channel_last = data_format == "NHWC";
    // update padding and dilation
    auto src_dims = input.dims();
    auto weights_dims = filter.dims();
    dnnl::memory::dim OH = (src_dims[2] - weights_dims[2] + paddings_t[0] + paddings_t[0]) / strides_t[0] + 1, // output height
                       OW = (src_dims[3] - weights_dims[3] + paddings_t[1] + paddings_t[1]) / strides_t[1] + 1; // output width
    dnnl::memory::dims dst_dims = {src_dims[0],weights_dims[0],OH,OW};
    //auto dst_dims = output->dims();

    //show_kernel("Conv2dOneDNN()" << " type=" << dnn_support::type2String<T>::name());
    show_kernel("Conv2dOneDNN()" << " type=" << dnn_support::type2String<T>::name()<<"; input dim = "<<src_dims<<"; out dim"<<dst_dims);

    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    auto* q = static_cast<sycl::queue*>(const_cast<void*>(ctx.stream()));
    auto eng =
        dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
    auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

    // Source (src), weights and destination (dst) tensors dimensions.
    auto src_data = input.data<T>();
    auto weights_data = filter.data<T>();
    auto out_data = ctx.template Alloc<T>(output);

    auto tensor_format_nchw = dnnl::memory::format_tag::nchw;
    auto tensor_format_oihw = dnnl::memory::format_tag::oihw;

    auto conv_src_md = dnnl::memory::desc(src_dims, dnn_support::toDnnType<T>::type, tensor_format_nchw);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dnn_support::toDnnType<T>::type, tensor_format_oihw);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dnn_support::toDnnType<T>::type, tensor_format_nchw);

    auto conv_src_mem = dnnl::memory(conv_src_md, eng, src_data);
    auto conv_weights_mem = dnnl::memory(conv_weights_md, eng, weights_data);
    auto conv_dst_mem = dnnl::memory(conv_dst_md, eng, out_data);
    
    // Create primitive descriptor.
    auto conv_pd = dnnl::convolution_forward::primitive_desc(eng,
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        conv_src_md, conv_weights_md, conv_dst_md,
        strides_dims, 
        //dilation_dims,
        padding_dims_l, padding_dims_r);

    // Create the primitive.
    auto conv_prim = dnnl::convolution_forward(conv_pd);

    // Primitive arguments.
    std::unordered_map<int, dnnl::memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    //conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});
    std::cout<<"before execute"<<std::endl;

    // Primitive execution: convolution with ReLU.
    conv_prim.execute(engine_stream, conv_args);

    engine_stream.wait();
    std::cout<<"execute end"<<std::endl;
 
}

}  // namespace custom_kernel


PD_BUILD_PHI_KERNEL(conv2d,
                    SYCL,
                    ALL_LAYOUT,
                    custom_kernel::Conv2dKernel,
                    float,
                    phi::dtype::float16) {}
