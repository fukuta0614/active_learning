#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
VGG16 model for chainer 1.8
"""

import chainer
import chainer.functions as F
import chainer.links as L
import os, sys
import numpy as np

shared = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(shared)
from ..functions import power_normalize, global_average_pooling_2d


class VGG16(chainer.Chain):
    """Single-GPU VGGNet(16layers)"""

    insize = 224

    def __init__(self, num_class=1000, texture=False, texture_layer='pool4', cbp=False, normalize=True, mil=False):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, pad=1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, pad=1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, pad=1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, pad=1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, pad=1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, pad=1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, pad=1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, pad=1)
            if not texture:
                self.conv5_1 = L.Convolution2D(512, 512, 3, pad=1)
                self.conv5_2 = L.Convolution2D(512, 512, 3, pad=1)
                self.conv5_3 = L.Convolution2D(512, 512, 3, pad=1)
                self.fc6 = L.Linear(25088, 4096)
                self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, num_class)

        self.texture = texture
        self.texture_layer = texture_layer
        self.cbp = cbp
        self.normalize = normalize
        self.mil = mil

    def forward(self, x):
        """
        h1 : (1, 64, 112, 112)
        h2 : (1, 128, 56, 56)
        h3 : (1, 256, 28, 28)
        h4 : (1, 512, 14, 14)
        h5 : (1, 512, 7, 7)

        :param x:
        :return:
        """
        h = x
        h = F.relu((self.conv1_1(h)))
        h = F.relu((self.conv1_2(h)))
        pool1 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu((self.conv2_1(pool1)))
        h = F.relu((self.conv2_2(h)))
        pool2 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu((self.conv3_1(pool2)))
        h = F.relu((self.conv3_2(h)))
        h = F.relu((self.conv3_3(h)))
        pool3 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu((self.conv4_1(pool3)))
        h = F.relu((self.conv4_2(h)))
        h = F.relu((self.conv4_3(h)))
        pool4 = F.max_pooling_2d(h, 2, stride=2)

        if self.texture:
            h = {'pool1': pool1, 'pool2': pool2, 'pool3': pool3, 'pool4': pool4}[self.texture_layer]
            if self.cbp:
                h = F.convolution_2d(h, self.W1) * F.convolution_2d(h, self.W2)
                h = global_average_pooling_2d(h)
                if self.normalize:
                    h = power_normalize(h)
                    h = F.normalize(h)

                h = self.fc8(F.dropout(h, 0.2))
                return h
            else:
                b, ch, height, width = h.data.shape
                h = F.reshape(h, (b, ch, width * height))
                h = F.batch_matmul(h, h, transb=True) / self.xp.float32(width * height)
                h = self.fc8(F.dropout(h, 0.4))
                return h
        else:
            h = F.relu((self.conv5_1(pool4)))
            h = F.relu((self.conv5_2(h)))
            h = F.relu((self.conv5_3(h)))
            h = F.max_pooling_2d(h, 2, stride=2)
            h = F.dropout(F.relu(self.fc6(h)), ratio=0.5)
            h = F.dropout(F.relu(self.fc7(h)), ratio=0.5)

            h = self.fc8(h)
            return h

    def load_pretrained(self, pretrained_path, num_class):
        chainer.serializers.load_npz(pretrained_path, self)
        self.convert_to_finetune_model(num_class)

    def convert_to_finetune_model(self, num_class):
        if self.cbp:
            randweight = np.load(os.path.join(shared, 'cbp/randweight_512_to_4096.npz'))
            self.add_persistent('W1', randweight['W1'])
            self.add_persistent('W2', randweight['W2'])
        self.fc8 = L.Linear(None, num_class)
        return

    def __call__(self, x, t):
        self.y = self.forward(x)
        self.loss = F.softmax_cross_entropy(self.y, t)
        self.accuracy = F.accuracy(self.y, t)
        return self.loss


class VGG16Feature(chainer.Chain):
    """Single-GPU VGGNet(16layers)"""

    insize = 224

    def __init__(self, input_ch=3):
        super(VGG16Feature, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(input_ch, 64, 3, pad=1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, pad=1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, pad=1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, pad=1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, pad=1)

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, x):
        """
        h1 : (1, 64, 112, 112)
        h2 : (1, 128, 56, 56)
        h3 : (1, 256, 28, 28)
        h4 : (1, 512, 14, 14)
        h5 : (1, 512, 7, 7)

        :param x:
        :param stop_at_final_conv:
        :param conv5_fc7:
        :return:
        """
        h = x
        h = F.relu((self.conv1_1(h)))
        h = F.relu((self.conv1_2(h)))
        h1 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu((self.conv2_1(h1)))
        h = F.relu((self.conv2_2(h)))
        h2 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu((self.conv3_1(h2)))
        h_ = F.relu((self.conv3_2(h)))
        h = F.relu((self.conv3_3(h_)))
        h3 = F.max_pooling_2d(h, 2, stride=2)
        return h2, h_, h3

    def __call__(self, x, t):
        self.clear()
        h = self.forward(x)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss
