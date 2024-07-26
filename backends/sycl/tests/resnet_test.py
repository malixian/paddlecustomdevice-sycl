import paddle
from paddle.vision.models import wide_resnet50_2

# build model
model = wide_resnet50_2()

# build model and load imagenet pretrained weight
model = wide_resnet50_2(pretrained=True)

x = paddle.rand([1, 3, 224, 224])
out = model(x)

print(out.shape)
