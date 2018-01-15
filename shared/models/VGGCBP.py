#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import VGG16_conv
import VGG19_conv
import numpy as np
from copy_model import copy_model
import os
import sys

pdim = 4096


def ssq(x):
    # pos =  F.clip(x,0.0,100000000.0)
    # neg = F.clip(-x,0.0,100000000.0)
    pos = F.relu(x)
    neg = F.relu(-x)
    return F.sqrt(pos) - F.sqrt(neg)


class VGG(chainer.Chain):
    def __init__(self, nclass=10, arch='VGG16', pooling='avg', p=5, a=0.25, svmpath=None, l2normalize=False):
        super(VGG, self).__init__()

        if arch == 'VGG16':
            VGGNet = VGG16_conv.VGGNet
            srcpath = '/data/unagi0/mukuta/VGG/pretrained/VGG_ILSVRC_16_layers.caffemodel'
            srcchainer = '/data/unagi0/mukuta/VGG/pretrained/VGG_ILSVRC_16_layers_cbp_conv.chainer'
        elif arch == 'VGG19':
            VGGNet = VGG19_conv.VGGNet
            srcpath = '/data/unagi0/mukuta/VGG/pretrained/VGG_ILSVRC_19_layers.caffemodel'
            srcchainer = '/data/unagi0/mukuta/VGG/pretrained/VGG_ILSVRC_19_layers_cbp_conv.chainer'
        else:
            print('error in arch')
        self.add_link('conv', VGGNet())
        if not (os.path.exists(srcchainer)):
            from chainer.links import caffe
            srcmodel = caffe.CaffeFunction(srcpath)
            copy_model(srcmodel, self.conv)
            chainer.serializers.save_npz(srcchainer, self.conv)
        else:
            chainer.serializers.load_npz(srcchainer, self.conv)

        if pooling == 'avg':
            self.add_link('fc', FC_avg(nclass, l2normalize))
        elif pooling == 'spp':
            self.add_link('fc', FC_spp(nclass, p, l2normalize))
        elif pooling == 'kweight':
            self.add_link('fc', FC_kweight(nclass, p, a, l2normalize))
        else:
            print('error in pooling')

        if svmpath:
            svm = np.load(svmpath)
            self.fc.fc.W.data = svm['coef_'].astype(np.float32)
            self.fc.fc.b.data = np.squeeze(svm['intercept_']).astype(np.float32)

        self.train = True

    def __call__(self, x, t):
        h = self.conv(x)
        h = self.fc(h, train=self.train)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class FC_avg(chainer.Chain):
    def __init__(self, nclass, l2normalize):
        super(FC_avg, self).__init__(fc=L.Linear(pdim, nclass))
        self.l2normalize = l2normalize
        randweight = np.load('cbp/randweight.npz')
        self.add_persistent('W1', randweight['W1'])
        self.add_persistent('W2', randweight['W2'])

    def feat(self, h):
        h = F.convolution_2d(h, self.W1) * F.convolution_2d(h, self.W2)
        h = F.average_pooling_2d(h, 28, 28)
        if self.l2normalize:
            h = F.reshape(h, (h.data.shape[0], -1))
            h = ssq(h)
            h = F.normalize(h)
        return h

    def __call__(self, h, train):
        h = self.feat(h)
        h = self.fc(h)
        return h


class FC_spp(chainer.Chain):
    def __init__(self, nclass, p, l2normalize):
        self.p = p
        psizes = [1, 5, 21]
        if p > 3 or p < 1:
            print('p out of range')
        else:
            psize = psizes[p - 1]
        super(FC_spp, self).__init__(fc=L.Linear(pdim * psize, nclass))
        self.l2normalize = l2normalize
        randweight = np.load('cbp/randweight.npz')
        self.add_persistent('W1', randweight['W1'])
        self.add_persistent('W2', randweight['W2'])

    def feat(self, h):
        h = F.convolution_2d(h, self.W1) * F.convolution_2d(h, self.W2)
        h = F.spatial_pyramid_pooling_2d(h, self.p, F.MaxPooling2D)
        if self.l2normalize:
            h = F.reshape(h, (h.data.shape[0], -1))
            h = ssq(h)
            h = F.normalize(h)
        return h

    def __call__(self, h, train):
        h = self.feat(h)
        h = self.fc(h)
        return h


class FC_kweight(chainer.Chain):
    def __init__(self, nclass, p, a, l2normalize):
        psize = (p + 1) * (p + 2) / 2
        super(FC_kweight, self).__init__(fc=L.Linear(pdim * psize, nclass))
        import scipy.io
        self.add_persistent('loadedfilter', np.reshape(
            scipy.io.loadmat('pyramid/kweight_%f_p%d.mat' % (a, p))['weights'].T.astype(np.float32) / (28. * 28),
            (1, 28 * 28, psize)))
        self.l2normalize = l2normalize
        randweight = np.load('cbp/randweight.npz')
        self.add_persistent('W1', randweight['W1'])
        self.add_persistent('W2', randweight['W2'])

    def feat(self, h, train=True):
        batchsize = h.data.shape[0]
        h = F.convolution_2d(h, self.W1) * F.convolution_2d(h, self.W2)
        h = F.batch_matmul(F.reshape(h, (batchsize, pdim, 28 * 28)),
                           chainer.Variable(chainer.cuda.cupy.tile(self.loadedfilter, (batchsize, 1, 1)),
                                            volatile='off' if train else 'on'))
        if self.l2normalize:
            h = F.reshape(h, (h.data.shape[0], -1))
            h = ssq(h)
            h = F.normalize(h)
        return h

    def __call__(self, h, train):
        h = self.feat(h, train)
        h = self.fc(h)
        return h
