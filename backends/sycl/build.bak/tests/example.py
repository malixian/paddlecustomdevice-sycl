# check the plugin status
import paddle
print(paddle.device.get_all_custom_device_type())

paddle.set_device("SYCL")

'''
x = paddle.to_tensor([1])
print(x)

x = x + x
print(x)
'''
paddle.enable_static()

input = paddle.static.data(name="input", shape=[12, 10], dtype="float32")
out = paddle.mean(input)

print(out)