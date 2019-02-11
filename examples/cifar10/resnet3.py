
from __future__ import print_function
from builtins import zip

from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet

# reference, resnet singa cifar 10 & pytorch architecture only
# 
# 
conv_bias = False

def conv(net, prefix, n, ksize, stride=1, pad=0, bn=True, relu=True, src=None):
    '''Add a convolution layer and optionally a batchnorm and relu layer.
    Args:
        prefix, a string for the prefix of the layer name
        n, num of filters for the conv layer
        bn, if true add batchnorm
        relu, if true add relu
    Returns:
        the last added layer
    '''
    ret = net.add(layer.Conv2D(
        prefix + '-conv', n, ksize, stride, pad=pad), src)
    if bn:
        ret = net.add(layer.BatchNormalization(prefix + '-bn'))
    if relu:
        ret = net.add(layer.Activation(prefix + '-relu'))
    return ret

def shortcut(net, prefix, inplane, outplane, stride, src, bn=False):
    '''Add a conv shortcut layer if inplane != outplane; or return the source
    layer directly.
    Args:
        prefix, a string for the prefix of the layer name
        bn, if true add a batchnorm layer after the conv layer
    Returns:
        return the last added layer or the source layer.
    '''
    if inplane == outplane:
        return src
    return conv(net, prefix + '-shortcut', outplane, 1, stride, 0, bn, False, src)

def Block(net, name, nb_filters, stride,expand=False):
    split = net.add(layer.Split(name + "-split", 2))
    if stride > 1 or expand:
        net.add(layer.Conv2D(name + "-br2-conv", nb_filters, 1, stride, pad=0), split)
        br2bn = net.add(layer.BatchNormalization(name + "-br2-bn"))
    net.add(layer.Conv2D(name + "-br1-conv1", nb_filters, 3, stride, pad=1), split)
    net.add(layer.BatchNormalization(name + "-br1-bn1"))
    net.add(layer.Activation(name + "-br1-relu"))
    net.add(layer.Conv2D(name + "-br1-conv2", nb_filters, 3, 1, pad=1))
    br1bn2 = net.add(layer.BatchNormalization(name + "-br1-bn2"))
    if stride > 1 or expand:
        net.add(layer.Merge(name + "-merge"), [br1bn2, br2bn])
    else:
        net.add(layer.Merge(name + "-merge"), [br1bn2, split])

def stage(sid, net, num_blk, inplane, midplane, outplane, stride, block, preact=False, add_bn=False):
    tempStr = "stage"+str(sid)+"-blk0"
    block(tempStr, net, inplane, midplane, outplane, stride, preact, add_bn)
    for i in range(1, num_blk):
      tempStr = "stage"+str(sid)+"-blk"+str(i)
      block(tempStr, net, outplane, midplane, outplane, 1, preact, add_bn)

def bottleneck(name,net,inplane, midplane,outplane,stride=1,preact=False, add_bn=False):
  split = net.add(layer.Split(name + '-split', 2))
  if preact:
      net.add(layer.BatchNormalization(name + '-preact-bn'))
      net.add(layer.Activation(name + '-preact-relu'))
  conv(net, name + '-0', midplane, 1, 1, 0, True, True)
  conv(net, name + '-1', midplane, 3, stride, 1, True, True)
  br0 = conv(net, name + '-2', outplane, 1, 1, 0, not (preact or add_bn), False)
  br1 = shortcut(net, name, inplane, outplane, stride, split, not add_bn)
  ret = net.add(layer.Merge(name + '-add'), [br0, br1])
  if add_bn:
      ret = net.add(layer.BatchNormalization(name + '-add-bn'))
  if not preact:
      ret = net.add(layer.Activation(name + '-add-relu'))
  return ret

def Block_dep(net, name, nb_filters, stride):
    split = net.add(layer.Split(name + "-split", 2))
    if stride > 1:
        net.add(layer.Conv2D(name + "-br2-conv", nb_filters, 1, stride, pad=0), split)
        br2bn = net.add(layer.BatchNormalization(name + "-br2-bn"))
    net.add(layer.Conv2D(name + "-br1-conv1", nb_filters, 3, stride, pad=1), split)
    net.add(layer.BatchNormalization(name + "-br1-bn1"))
    net.add(layer.Activation(name + "-br1-relu1"))

    net.add(layer.Conv2D(name + "-br1-conv2", nb_filters, 3, 1, pad=1))
    net.add(layer.BatchNormalization(name + "-br1-bn2"))
    net.add(layer.Activation(name + "-br1-relu2"))

    net.add(layer.Conv2D(name + "-br1-conv3", nb_filters, 3, 1, pad=1))
    br1bn2 = net.add(layer.BatchNormalization(name + "-br1-bn3"))
    if stride > 1:
        net.add(layer.Merge(name + "-merge"), [br1bn2, br2bn])
    else:
        net.add(layer.Merge(name + "-merge"), [br1bn2, split])

cfg = { 18: [2, 2, 2, 2],  # basicblock
    34: [3, 4, 6, 3],  # basicblock
    50: [3, 4, 6, 3],  # bottleneck
    101: [3, 4, 23, 3], # bottleneck
    152: [3, 8, 36, 3], # bottleneck
    200: [3, 24, 36, 3]} # bottleneck

def create_net(depth, use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'
    plane=64
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    net.add(layer.Conv2D("conv1", plane, 3, 1, pad=1, input_sample_shape=(3, 32, 32)))
    net.add(layer.BatchNormalization("bn1"))
    net.add(layer.Activation("relu1"))
    
    conf = cfg[depth]

    if depth <= 34:
      Block(net, "2a", plane, 1)
      for i in range(1,conf[0]):
        Block(net,'2%d' %i,plane,1)

      Block(net, "3a", plane, 1,expand=True)
      for i in range(1,conf[1]):
        Block(net,'3%d' %i,plane,1)
      
      Block(net, "4a", plane*2, 2)
      for i in range(1,conf[2]):
        Block(net,'4%d' %i,plane*2,1)

      Block(net, "5a", plane*4, 2,expand=True)
      for i in range(1,conf[3]):
        Block(net,'5%d' %i,plane*4,1)

    else:
      Block_dep(net, "2a", 64, 1)
      for i in range(1,conf[0]):
        Block_dep(net,'2%d' %i,64,1)

      Block_dep(net, "3a", 128, 1,expand=True)
      for i in range(1,conf[1]):
        Block_dep(net,'3%d' %i,128,1)

      Block_dep(net, "4a", 256, 2)
      for i in range(1,conf[2]):
        Block_dep(net,'4%d' %i,256,1)

      Block_dep(net, "5a", 512, 2)
      for i in range(1,conf[3]):
        Block_dep(net,'5%d' %i,512,1)
    #   stage(0, net, conf[0], 64, 64, 256, 1, bottleneck, add_bn=True)
    #   stage(1, net, conf[1], 256, 128, 512, 2, bottleneck, add_bn=True)
    #   stage(2, net, conf[2], 512, 256, 1024, 2, bottleneck, add_bn=True)
    #   stage(3, net, conf[3], 1024, 512, 2048, 2, bottleneck, add_bn=True)

    net.add(layer.AvgPooling2D("pool4", 8, 8, border_mode='valid'))
    #net.add(layer.AvgPooling2D("pool4", 8, 8, border_mode='valid'))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('ip5', 10))
    print('Start intialization............')
    for (p, name) in zip(net.param_values(), net.param_names()):
        # print name, p.shape
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv' in name:
                # initializer.gaussian(p, 0, math.sqrt(2.0/p.shape[1]))
                initializer.gaussian(p, 0, 9.0 * p.shape[0])
            else:
                initializer.uniform(p, p.shape[0], p.shape[1])
        else:
            p.set_value(0)
        # print name, p.l1()

    return net
