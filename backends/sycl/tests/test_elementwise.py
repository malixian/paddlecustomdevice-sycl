import paddle.fluid as fluid
import numpy as np
import paddle

def gen_data():
    return {
        "x": np.array([2, 3, 4]).astype('float32'),
        "y": np.array([1, 5, 2]).astype('float32')
    }
paddle.enable_static()
x = fluid.data(name="x", shape=[3], dtype='float32')
y = fluid.data(name="y", shape=[3], dtype='float32')
z = fluid.layers.elementwise_mul(x, y)
# z = x * y

place = paddle.CustomPlace("SYCL", 0)
exe = fluid.Executor(place)
z_value = exe.run(feed=gen_data(),
                    fetch_list=[z.name])

print(z_value) # [2., 15., 8.]
