from __future__ import print_function
from builtins import zip

from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet


def Block(net, name, nb_filters, stride):
    split = net.add(layer.Split(name + "-split", 2))
    if stride > 1:
        net.add(layer.Conv2D(name + "-br2-conv", nb_filters, 1, stride, pad=0), split)
        br2bn = net.add(layer.BatchNormalization(name + "-br2-bn"))
    net.add(layer.Conv2D(name + "-br1-conv1", nb_filters, 3, stride, pad=1), split)
    net.add(layer.BatchNormalization(name + "-br1-bn1"))
    net.add(layer.Activation(name + "-br1-relu"))
    net.add(layer.Conv2D(name + "-br1-conv2", nb_filters, 3, 1, pad=1))
    br1bn2 = net.add(layer.BatchNormalization(name + "-br1-bn2"))
    if stride > 1:
        net.add(layer.Merge(name + "-merge"), [br1bn2, br2bn])
    else:
        net.add(layer.Merge(name + "-merge"), [br1bn2, split])

def ublock(net,namei,slist):
    net.add(layer.Conv2D("conv"+str(namei), 16, 3, 1, pad=1))
    split = net.add(layer.Split("split"+str(namei), 2))
    slist.append(split)
    net.add(layer.BatchNormalization("bn"+str(namei)),split)
    ans = net.add(layer.Activation("relu"+str(namei)))
    return ans

def ublock2(net,namei,input0,input1):
    net.add(layer.Concat('concat'+str(namei),1), [input0, input1])
    net.add(layer.Conv2D("conv"+str(namei), 16, 3, 1, pad=1))
    net.add(layer.BatchNormalization("bn"+str(namei)))
    ans = net.add(layer.Activation("relu"+str(namei)))
    return ans

def create_net(use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'
    slist=[]
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    net.add(layer.Conv2D("conv1", 16, 3, 1, pad=1, input_sample_shape=(3, 32, 32)))
    net.add(layer.BatchNormalization("bn1"))
    net.add(layer.Activation("relu1"))

    ublock(net,2,slist)
    ublock(net,3,slist)
    b4 = ublock(net,4,slist)

    #slist.append(1)
    #slist.append(1)
    #slist.append(1)

    b5 = ublock2(net,5,slist[2],b4)
    b6 = ublock2(net,6,slist[1],b5)
    ublock2(net,7,slist[0],b6)



    net.add(layer.AvgPooling2D("pool4", 32, 32, border_mode='valid'))
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
