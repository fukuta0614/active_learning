import os, sys
import numpy as np
import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

shared = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(shared)
from ..functions import power_normalize, global_average_pooling_2d


class BuildingBlock(chainer.Chain):
    """A building block that consists of several Bottleneck layers.

    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW)
            self.forward = [self.a]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels, initialW)
                setattr(self, name, bottleneck)
                self.forward.append(bottleneck)

    def __call__(self, x):
        for l in self.forward:
            x = l(x)
        return x


class BottleneckA(chainer.Chain):
    """A bottleneck layer that reduces the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None):
        super(BottleneckA, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(chainer.Chain):
    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(in_channels)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h


class ResNet50(chainer.Chain):
    insize = 224

    def __init__(self, n_layers=50, num_class=1000, mil=False, texture=False, cbp=False, normalize=True):
        super(ResNet50, self).__init__()

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = BuildingBlock(block[0], 64, 64, 256, 1, initialW=initializers.HeNormal())
            self.res3 = BuildingBlock(block[1], 256, 128, 512, 2, initialW=initializers.HeNormal())
            if not texture:
                self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2, initialW=initializers.HeNormal())
                self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2, initialW=initializers.HeNormal())
            self.fc6 = L.Linear(2048, num_class)

        self.mil = mil
        self.texture = texture
        self.cbp = cbp
        self.normalize = normalize

    def forward(self, x):
        """
        res3 (1, 512, 28, 28)
        res4 (1, 1024, 14, 14)
        res5 (1, 2048, 7, 7)
        :param x:
        :param t:
        :return:
        """
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)

        if self.texture:
            if self.cbp:
                h = F.convolution_2d(h, self.W1) * F.convolution_2d(h, self.W2)
                h = global_average_pooling_2d(h)
                if self.normalize:
                    h = power_normalize(h)
                    h = F.normalize(h)
                self.cbp_feat = h
                h = self.fc6(F.dropout(h, 0.25))
                return h
            else:
                b, ch, height, width = h.data.shape
                h = F.reshape(h, (b, ch, width * height))
                h = F.batch_matmul(h, h, transb=True) / self.xp.float32(width * height)
                if self.normalize:
                    h = power_normalize(h)
                    h = F.normalize(h)
                h = self.fc6(F.dropout(h, 0.25))
                return h
        else:
            h = self.res4(h)
            h = self.res5(h)
            if self.mil:
                h = F.max(h, axis=0, keepdims=True)
            h = _global_average_pooling_2d(h)
            h = self.fc6(h)

            return h

    def load_pretrained(self, pretrained_path, num_class):
        chainer.serializers.load_npz(pretrained_path, self)
        if self.cbp:
            randweight = np.load(os.path.join(shared, 'cbp/randweight_512_to_4096.npz'))
            self.add_persistent('W1', randweight['W1'])
            self.add_persistent('W2', randweight['W2'])
        self.fc6 = L.Linear(None, num_class)

        return

    def __call__(self, x, t):
        self.y = self.forward(x)
        self.loss = F.softmax_cross_entropy(self.y, t)
        self.accuracy = F.accuracy(self.y, t)

        return self.loss
