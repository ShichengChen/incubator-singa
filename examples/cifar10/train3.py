

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
try:
    import pickle
except ImportError:
    import cPickle as pickle
import numpy as np
import os
import argparse

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2

#import resnet3 as resnet
import vgg3 as vgg

from datetime import datetime
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
def load_dataset(filepath):
    print('Loading data file %s' % filepath)
    with open(filepath, 'rb') as fd:
        cifar10 = pickle.load(fd)
    image = cifar10['data'].astype(dtype=np.uint8)
    image = image.reshape((-1, 3, 32, 32))
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    return image, label


def load_train_data(dir_path, num_batches=5):
    labels = []
    batchsize = 10000
    images = np.empty((num_batches * batchsize, 3, 32, 32), dtype=np.uint8)
    for did in range(1, num_batches + 1):
        fname_train_data = dir_path + "/data_batch_{}".format(did)
        image, label = load_dataset(fname_train_data)
        images[(did - 1) * batchsize:did * batchsize] = image
        labels.extend(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def load_test_data(dir_path):
    images, labels = load_dataset(dir_path + "/test_batch")
    return np.array(images,  dtype=np.float32), np.array(labels, dtype=np.int32)


def normalize_for_vgg(train_x, test_x):
    mean = train_x.mean()
    std = train_x.std()
    train_x -= mean
    test_x -= mean
    train_x /= std
    test_x /= std
    return train_x, test_x


def normalize_for_alexnet(train_x, test_x):
    mean = np.average(train_x, axis=0)
    train_x -= mean
    test_x -= mean
    return train_x, test_x


def vgg_lr(epoch):
    return 0.1 / float(1 << (epoch // 25))


def alexnet_lr(epoch):
    if epoch < 120:
        return 0.001
    elif epoch < 130:
        return 0.0001
    else:
        return 0.00001


def resnet_lr(epoch):
    if epoch < 81:
        return 0.1
    elif epoch < 122:
        return 0.01
    else:
        return 0.001


def caffe_lr(epoch):
    if epoch < 8:
        return 0.001
    else:
        return 0.0001


def train(data, net, max_epoch, get_lr, weight_decay, batch_size=100,
          use_cpu=False):
    print('Start intialization............')
    if use_cpu:
        print('Using CPU')
        dev = device.get_default_device()
    else:
        print('Using GPU')
        dev = device.create_cuda_gpu()

    net.to_device(dev)
    opt = optimizer.SGD(momentum=0.9, weight_decay=weight_decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)

    tx = tensor.Tensor((batch_size, 3, 32, 32), dev)
    ty = tensor.Tensor((batch_size,), dev, core_pb2.kInt)
    train_x, train_y, test_x, test_y = data
    num_train_batch = train_x.shape[0] // batch_size
    num_test_batch = test_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)
    fileTimeLog =open("epochTimeLog.text","a")
    duration_itr = []
    for epoch in range(1):
        np.random.shuffle(idx)
        loss, acc = 0.0, 0.0
        print('Epoch %d' % epoch)
        print(datetime.now().timetz()) # miliseconds
        print(int(round(time.time()*1000)))
        fileTimeLog.write('Epoch %d: ' % epoch)
        fileTimeLog.write(str(int(round(time.time()*1000))))
        fileTimeLog.write('\n')
        tic = datetime.now()
        for b in range(13): #num_train_batch):
            print ("start of iteration %d: " %b)
            #time.sleep(1)
            toc = datetime.now()
            duration_itr.append(toc-tic)
            tic = toc
            fileTimeLog.write('iteration %d: ' % b)
            fileTimeLog.write(str(int(round(time.time()*1000))))
            fileTimeLog.write('\n')
            x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
            y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
            print('numpy to singa tensor')
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            print('train netword forward')
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            print('backward network')
            for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s), b)
            print('loss,acc',loss,acc)
            # update progress bar
            utils.update_progress(b * 1.0 / num_train_batch,
                                  'training loss = %f, accuracy = %f' % (l, a))
        info = '\ntraining loss = %f, training accuracy = %f, lr = %f' \
               % ((loss / num_train_batch), (acc / num_train_batch), get_lr(epoch))
        print(info)
        print ("now prints duration")
        for itm in duration_itr:
            print (itm)

        loss, acc = 0.0, 0.0
        for b in range(0):
            x = test_x[b * batch_size: (b + 1) * batch_size]
            y = test_y[b * batch_size: (b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            acc += a

        print('test loss = %f, test accuracy = %f' %
              ((loss / num_test_batch), (acc / num_test_batch)))
    fileTimeLog.close()
    #net.save('model', 20)  # save model params into checkpoint file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dcnn for cifar10')
    parser.add_argument('model', choices=['vgg', 'alexnet', 'resnet', 'caffe'],
                        default='alexnet')
    parser.add_argument('data', default='cifar-10-batches-py')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('batch_size',type=int, default=100)
    parser.add_argument('depth',type=int)
    args = parser.parse_args()
    assert os.path.exists(args.data), \
        'Pls download the cifar10 dataset via "download_data.py py"'
    print('Loading data ..................')
    train_x, train_y = load_train_data(args.data)
    test_x, test_y = load_test_data(args.data)
    if args.model == 'caffe':
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = caffe_net.create_net(args.use_cpu)
        # for cifar10_full_train_test.prototxt
        train((train_x, train_y, test_x, test_y), net, 160, alexnet_lr, 0.004,
              use_cpu=args.use_cpu,batch_size=args.batch_size)
        # for cifar10_quick_train_test.prototxt
        # train((train_x, train_y, test_x, test_y), net, 18, caffe_lr, 0.004,
        #      use_cpu=args.use_cpu)
    elif args.model == 'alexnet':
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = alexnet.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 2, alexnet_lr, 0.004,
              use_cpu=args.use_cpu,batch_size=args.batch_size)
    elif args.model == 'vgg':
        train_x, test_x = normalize_for_vgg(train_x, test_x)
        depth = args.depth
        net = vgg.create_net(depth,args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 250, vgg_lr, 0.0005,
              use_cpu=args.use_cpu,batch_size=args.batch_size)
    else:
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        depth = args.depth
        net = resnet.create_net(depth,args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 200, resnet_lr, 1e-4,
              use_cpu=args.use_cpu,batch_size=args.batch_size)