import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import Uniform


class ConvBNReLU(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 initialW=chainer.initializers.HeNormal(), nobias=True):
        super(ConvBNReLU, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize, stride, pad,
                                        initialW=initialW, nobias=nobias)
            self.bn = L.BatchNormalization(out_channels, eps=1e-5)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)

        return F.relu(h)


class ConvNet(chainer.Chain):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        with self.init_scope():
            self.conv11 = ConvBNReLU(3, 64, 3, pad=1)
            self.conv12 = ConvBNReLU(64, 64, 3, pad=1)
            self.conv21 = ConvBNReLU(64, 128, 3, pad=1)
            self.conv22 = ConvBNReLU(128, 128, 3, pad=1)
            self.conv31 = ConvBNReLU(128, 256, 3, pad=1)
            self.conv32 = ConvBNReLU(256, 256, 3, pad=1)
            self.conv33 = ConvBNReLU(256, 256, 3, pad=1)
            self.conv34 = ConvBNReLU(256, 256, 3, pad=1)
            self.fc4 = L.Linear(256 * 4 * 4, 1024, initialW=Uniform(1. / math.sqrt(256 * 4 * 4)))
            self.fc5 = L.Linear(1024, 1024, initialW=Uniform(1. / math.sqrt(1024)))
            self.fc6 = L.Linear(1024, n_classes, initialW=Uniform(1. / math.sqrt(1024)))

    def forward(self, x):
        h = self.conv11(x)
        h = self.conv12(h)
        h = F.max_pooling_2d(h, 2)

        h = self.conv21(h)
        h = self.conv22(h)
        h = F.max_pooling_2d(h, 2)

        h = self.conv31(h)
        h = self.conv32(h)
        h = self.conv33(h)
        h = self.conv34(h)
        h = F.max_pooling_2d(h, 2)

        h = F.dropout(F.relu(self.fc4(h)))
        h = F.dropout(F.relu(self.fc5(h)))

        return self.fc6(h)

    def __call__(self, x, t):
        self.y = self.forward(x)
        self.loss = F.softmax_cross_entropy(self.y, t)
        self.acc = F.accuracy(self.y, t)
        return self.loss
