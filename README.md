# 基于PaddleCustomDevice接入oneDNN

# 部署方式

## 编译LLVM-SYCL 编译器
- LLVM-SYCL是SYCL编程模型的编译器，oneDNN需要依赖于LLVM-SYCL。
- LLVM-SYCL版本依赖于intel/llvm的llvm-nightly-2023-01-31分支
- 编译方式：
  \`\`\`
  cd llvm-sycl
  mkdir build
  cd build
  python3 ../buildbot/configure.py --hip
  python3 ../buildbot/compile.py
  \`\`\`

## 编译oneDNN
- 跨平台的统一DNN库，目前主要针对海光DCU进行适配

\`\`\`
export CC=clang
export CXX=clang++

mkdir build
cd build
cmake -DDNNL_CPU_RUNTIME=TBB -DDNNL_GPU_RUNTIME=DPCPP -DCMAKE_BUILD_TYPE=Release \
      -DONEDNN_BUILD_TESTS=OFF -DDNNL_GPU_VENDOR=AMD -DONEDNN_BUILD_GRAPH=OFF -G Ninja ..
#ninja -j 16
\`\`\`

## 编译PaddleCustomDevice
- [飞桨自定义接入硬件后端(SYCL)](backends/SYCL/README.md)


# PaddleCustomDevice

简体中文 | [English](./README_en.md) | [日本語](./README_ja.md)

『飞桨』自定义硬件接入实现。

## 使用指南

方案设计参考[Custom Device 接入方案介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/custom_device_overview_cn.html)，开发指南请参考[新硬件接入示例](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/custom_device_example_cn.html)且示例代码位于 [CustomCPU](backends/custom_cpu/README_cn.md)。

## 硬件后端

飞桨自定义硬件接入支持如下硬件后端：

- [飞桨自定义接入硬件后端(昇腾NPU)](backends/npu/README_cn.md)
- [飞桨自定义接入硬件后端(寒武纪MLU)](backends/mlu/README_cn.md)
- [飞桨自定义接入硬件后端(英特尔GPU)](backends/SYCL/README.md)
- [飞桨自定义接入硬件后端(苹果MPS)](backends/mps/README.md)
- [飞桨自定义接入硬件后端(壁仞GPU)](backends/biren_gpu/README_cn.md)
- [飞桨自定义接入硬件后端(燧原GCU)](backends/gcu/README_cn.md)

## 版权和许可证

PaddleCustomDevice由[Apache-2.0 license](LICENSE)提供。
