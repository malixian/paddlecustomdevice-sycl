# PaddlePaddle Custom Device Implementaion for Custom Intel GPU

Please refer to the following steps to compile, install and verify the custom device implementaion for Custom Intel GPU.

## Activate oneapi env vars

```bash
source load.sh
```

## Get Sources

```bash
# clone source
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# get the latest submodule source code
git submodule sync
git submodule update --remote --init --recursive
```

## Compile and Install

```bash
# navigate to implementaion for Custom CPU
cd backends/SYCL

# before compiling, ensure that Paddle is installed, you can run the following command
pip install paddlepaddle==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# create the build directory and navigate in
mkdir build && cd build

cmake ..
make -j $(nproc)

# using pip to install the output
pip install dist/paddle_custom_SYCL*.whl
```

## Verification

```bash
# check the plugin status
python3.9 -c "import paddle; print('SYCL' in paddle.device.get_all_custom_device_type())"

# expected output
True

```
