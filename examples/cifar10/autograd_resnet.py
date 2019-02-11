#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# the code is modified from
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

from singa import autograd
from singa import tensor
from singa import device
from singa import opt
import time
import numpy as np
from tqdm import trange
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return autograd.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1)


class BasicBlock(autograd.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = autograd.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = autograd.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.rand = 1

    def __call__(self, x):
        residual = x
        rand = np.random.uniform(0,1)
        #if(rand<0.5):return self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = autograd.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = autograd.add(out, residual)
        out = autograd.relu(out)

        return out


class Bottleneck(autograd.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = autograd.Conv2d(
            inplanes, planes, kernel_size=1)
        self.bn1 = autograd.BatchNorm2d(planes)
        self.conv2 = autograd.Conv2d(planes, planes, kernel_size=3,
                                     stride=stride,
                                     padding=1)
        self.bn2 = autograd.BatchNorm2d(planes)
        self.conv3 = autograd.Conv2d(
            planes, planes * self.expansion, kernel_size=1)
        self.bn3 = autograd.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = autograd.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = autograd.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = autograd.add(out, residual)
        out = autograd.relu(out)

        return out


class ResNet(autograd.Layer):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        inp=16
        self.conv1 = autograd.Conv2d(3, inp, kernel_size=3, stride=1, padding=1)
        self.bn1 = autograd.BatchNorm2d(inp)
        self.layer1 = self._make_layer( inp,inp, layers[0])
        self.layer2 = self._make_layer( inp,inp*2, layers[1], stride=2)
        self.layer3 = self._make_layer( inp*2,inp*4, layers[2], stride=2)
        self.layer4 = self._make_layer( inp*4,inp*4, layers[3], stride=1)
        self.avgpool = autograd.AvgPool2d(8, stride=8)
        self.fc = autograd.Linear(inp*4, num_classes)

    def _make_layer(self,  inp,outp, blocks, stride=1):
        downsample = None
        if stride != 1 or inp!=outp:
            conv = autograd.Conv2d(inp, outp,kernel_size=1, stride=stride)
            bn = autograd.BatchNorm2d(outp)

            def downsample(x):
                return bn(conv(x))

        layers = []
        layers.append(BasicBlock(inp, outp, stride, downsample))
        for i in range(blocks):
            layers.append(BasicBlock(outp, outp))

        def forward(x):
            for layer in layers:
                x = layer(x)
            return x
        return forward

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = autograd.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = autograd.flatten(x)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


if __name__ == '__main__':
    model = resnet18()
    print('Start intialization............')
    #dev = device.create_cuda_gpu_on()
    dev = device.create_cuda_gpu()
    niters = 200
    niters = 13
    batch_size = 200
    IMG_SIZE = 32
    sgd = opt.SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)

    tx = tensor.Tensor((batch_size, 3, IMG_SIZE, IMG_SIZE), dev)
    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    autograd.training = True
    train_x=np.random.randn(batch_size*batch_size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    train_y = np.random.randint(0, 10, batch_size*batch_size, dtype=np.int32)
    idx = np.arange(train_x.shape[0], dtype=np.int32)
    for i in np.arange(niters):
        time.sleep(1)
        np.random.shuffle(idx)
        x = train_x[idx[i * batch_size: (i + 1) * batch_size]]
        y = train_y[idx[i * batch_size: (i + 1) * batch_size]]
        #print('numpy to singa tensor')
        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)

        print('start forward')
        output = model(tx)
        loss = autograd.softmax_cross_entropy(output, ty)
        print('start backward')
        for p, g in autograd.backward(loss):
            sgd.update(p, g)
        print('end iteration',i)
