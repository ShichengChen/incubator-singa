
#####

from __future__ import print_function
from builtins import zip

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))

from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet

ffnet.verbose=True

def ConvBnReLU(net, name, nb_filers, sample_shape=None,recompute=1):
    net.add(layer.Conv2D(name + '_1', nb_filers, 3, 1, pad=1,
                         input_sample_shape=sample_shape))
    if(recompute == 0):
        net.add(layer.BatchNormalization(name + '_2'))
        net.add(layer.Activation(name + '_3'))
    else:
        cur = layer.BatchNormalizationWithReLu(name+'_2_3')
        net.add(cur)
        return cur

def create_net(depth,use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    # part 1
    recompute=0
    if depth != 11:
        ConvBnReLU(net, 'conv1_1', 64, (3, 32, 32),recompute=recompute)
        net.add(layer.Dropout('drop1', 0.3))
        cur1 = ConvBnReLU(net, 'conv1_2', 64,recompute=recompute)
        if(recompute==0 or recompute==1):
            net.add(layer.MaxPooling2D('pool1', 2, 2, border_mode='valid'))
        else:
            net.add(layer.reMaxPooling2D('pool1', 2, 2, border_mode='valid',pre=cur1,reflag=recompute))
    else:
        cur1 = ConvBnReLU(net, 'conv1_2', 64, (3, 32, 32),recompute=recompute)
        if(recompute==0 or recompute==1):
            net.add(layer.MaxPooling2D('pool1', 2, 2, border_mode='valid'))
        else:
            net.add(layer.reMaxPooling2D('pool1', 2, 2, border_mode='valid',pre=cur1,reflag=recompute))
    #part 2
    if depth != 11:
        ConvBnReLU(net, 'conv2_1', 128,recompute=recompute)
        net.add(layer.Dropout('drop2_1', 0.4))
    cur2 = ConvBnReLU(net, 'conv2_2', 128,recompute=recompute)
    if(recompute==0 or recompute==1):
        net.add(layer.MaxPooling2D('pool2', 2, 2, border_mode='valid'))
    else:
        net.add(layer.reMaxPooling2D('pool2', 2, 2, border_mode='valid',pre=cur2,reflag=recompute))
    #part 3
    if depth == 16:
        ConvBnReLU(net, 'conv3_0', 256,recompute=recompute)
        net.add(layer.Dropout('drop3_1', 0.4))
    if depth == 19:
        ConvBnReLU(net, 'conv3_0', 256,recompute=recompute)
        net.add(layer.Dropout('drop3_0', 0.4))
        ConvBnReLU(net, 'conv3_1', 256,recompute=recompute)
        net.add(layer.Dropout('drop3_1', 0.4))
    ConvBnReLU(net, 'conv3_2', 256,recompute=recompute)
    net.add(layer.Dropout('drop3_2', 0.4))
    cur3 = ConvBnReLU(net, 'conv3_3', 256,recompute=recompute)
    if(recompute==0 or recompute==1):
        net.add(layer.MaxPooling2D('pool3', 2, 2, border_mode='valid'))
    else:
        net.add(layer.reMaxPooling2D('pool3', 2, 2, border_mode='valid',pre=cur3,reflag=recompute))
    # part 4
    if depth == 16:
        ConvBnReLU(net, 'conv4_0', 512,recompute=recompute)
        net.add(layer.Dropout('drop4_1', 0.4))
    if depth == 19:
        ConvBnReLU(net, 'conv4_0', 512,recompute=recompute)
        net.add(layer.Dropout('drop4_0', 0.4))
        ConvBnReLU(net, 'conv4_1', 512,recompute=recompute)
        net.add(layer.Dropout('drop4_1', 0.4))
    ConvBnReLU(net, 'conv4_2', 512,recompute=recompute)
    net.add(layer.Dropout('drop4_2', 0.4))
    cur4=ConvBnReLU(net, 'conv4_3', 512,recompute=recompute)
    #for i in range(10):ConvBnReLU(net, 'conv4_3_'+str(i), 512,recompute)
    if(recompute==0 or recompute==1):
        net.add(layer.MaxPooling2D('pool4', 2, 2, border_mode='valid'))
    else:
        net.add(layer.reMaxPooling2D('pool4', 2, 2, border_mode='valid',pre=cur4,reflag=recompute))
    # part 5
    if depth == 16:
        ConvBnReLU(net, 'conv5_0', 512,recompute=recompute)
        net.add(layer.Dropout('drop5_1', 0.4))
    if depth == 19:
        ConvBnReLU(net, 'conv5_0', 512,recompute=recompute)
        net.add(layer.Dropout('drop5_0', 0.4))
        ConvBnReLU(net, 'conv5_1', 512,recompute=recompute)
        net.add(layer.Dropout('drop5_1', 0.4))
    ConvBnReLU(net, 'conv5_2', 512,recompute=0)
    net.add(layer.Dropout('drop5_2', 0.4))
    ConvBnReLU(net, 'conv5_3', 512,recompute=0)
    net.add(layer.MaxPooling2D('pool5', 2, 2, border_mode='valid'))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dropout('drop_flat', 0.5))
    net.add(layer.Dense('ip1', 512))
    net.add(layer.BatchNormalization('batchnorm_ip1'))
    net.add(layer.Activation('relu_ip1'))
    net.add(layer.Dropout('drop_ip2', 0.5))
    net.add(layer.Dense('ip2', 10))
    #print('Start intialization............')
    for (p, name) in zip(net.param_values(), net.param_names()):
        #print(name, p.shape)
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv' in name:
                initializer.gaussian(p, 0, 3 * 3 * p.shape[0])
            else:
                p.gaussian(0, 0.02)
        else:
            p.set_value(0)
        #print(name, p.l1())

    return net