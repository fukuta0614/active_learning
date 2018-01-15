import os

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import link
from .functions import bilinear_pooling, compact_bilinear_pooling
from . import CNN_ARCH_INFO

cbp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cbp')


class TextureRegularizedClassifier(link.Chain):
    def __init__(self, base_cnn, pretrained_model=None, texture_layer=None, num_class=1000):
        super(TextureRegularizedClassifier, self).__init__()
        cf = CNN_ARCH_INFO[base_cnn]
        cnn = cf['cnn'](pretrained_model=pretrained_model)
        self.logit = cf['logit']
        if base_cnn == 'googlenet' and pretrained_model is not None:
            self.logit = cf['logit'][:1]
        self.texture_layer = cf['default_texture_layer'] if texture_layer is None else texture_layer
        self.layers = self.logit + [self.texture_layer]

        if num_class != 1000:
            for layer in self.logit:
                setattr(cnn, layer, L.Linear(None, num_class))

        with self.init_scope():
            self.cnn = cnn

    def forward(self, x):
        act = self.cnn(x, layers=self.layers)
        h = act[self.logit[0]]
        return h

    def __call__(self, x, t):

        if len(x.shape) == 5:
            x1 = x[:, 0, :, :, :]
            x2 = x[:, 1, :, :, :]
            activations1 = self.cnn(x1, layers=self.layers)
            activations2 = self.cnn(x2, layers=self.layers)

            if len(self.logit) == 3:
                # for googlenet
                loss1_1, loss1_2, loss1_3 = [F.softmax_cross_entropy(activations1[layer], t) for layer in self.logit]
                loss2_1, loss2_2, loss2_3 = [F.softmax_cross_entropy(activations2[layer], t) for layer in self.logit]
                loss1 = 0.3 * (loss1_1 + loss1_2) + loss1_3
                loss2 = 0.3 * (loss2_1 + loss2_2) + loss2_3
            else:
                loss1 = F.softmax_cross_entropy(activations1[self.logit[0]], t)
                loss2 = F.softmax_cross_entropy(activations2[self.logit[0]], t)

            h = (activations1[self.logit[0]] + activations2[self.logit[0]]) / 2

            texture_feat1 = bilinear_pooling(activations1[self.texture_layer])
            texture_feat2 = bilinear_pooling(activations2[self.texture_layer])
            texture_loss = F.mean_squared_error(texture_feat1, texture_feat2)
            self.loss = loss1 + loss2 + texture_loss
        else:
            activations = self.cnn(x, layers=self.layers)
            if len(self.logit) == 3:
                loss1_1, loss1_2, loss1_3 = [F.softmax_cross_entropy(activations[layer], t) for layer in self.logit]
                loss = 0.3 * (loss1_1 + loss1_2) + loss1_3
            else:
                loss = F.softmax_cross_entropy(activations[self.logit[0]], t)
            self.loss = loss
            h = activations[self.logit[0]]

        self.accuracy = F.accuracy(h, t)
        chainer.report({'loss': self.loss, 'acc': self.accuracy}, self)
        return self.loss


class CBPRegularizedClassifier(link.Chain):
    def __init__(self, base_cnn, pretrained_model=None, texture_layer=None, cbp_size=4096, num_class=1000):
        super(CBPRegularizedClassifier, self).__init__()
        cf = CNN_ARCH_INFO[base_cnn]
        cnn = cf['cnn']()
        self.logit = cf['logit']
        if base_cnn == 'googlenet' and pretrained_model is not None:
            self.logit = cf['logit'][:1]
        self.texture_layer = cf['default_texture_layer'] if texture_layer is None else texture_layer
        self.layers = self.logit + [self.texture_layer]

        feat_size = cf['layer_info'][self.texture_layer]
        randweight = np.load(os.path.join(cbp_dir, 'randweight_{}_to_{}.npz'.format(feat_size, cbp_size)))

        if num_class != 1000:
            for layer in self.logit:
                setattr(cnn, layer, L.Linear(None, num_class))

        with self.init_scope():
            self.cnn = cnn
            self.add_persistent('W1', randweight['W1'])
            self.add_persistent('W2', randweight['W2'])

    def __call__(self, x, t):

        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]
        activations1 = self.cnn(x1, layers=self.layers)
        activations2 = self.cnn(x2, layers=self.layers)

        if len(self.logit) == 3:
            # for googlenet
            loss1_1, loss1_2, loss1_3 = [F.softmax_cross_entropy(activations1[layer], t) for layer in self.logit]
            loss2_1, loss2_2, loss2_3 = [F.softmax_cross_entropy(activations2[layer], t) for layer in self.logit]
            loss1 = 0.3 * (loss1_1 + loss1_2) + loss1_3
            loss2 = 0.3 * (loss2_1 + loss2_2) + loss2_3
        else:
            loss1 = F.softmax_cross_entropy(activations1[self.logit[0]], t)
            loss2 = F.softmax_cross_entropy(activations2[self.logit[0]], t)

        h = (activations1[self.logit[0]] + activations2[self.logit[0]]) / 2

        texture_feat1 = compact_bilinear_pooling(activations1[self.texture_layer], {'W1': self.W1, 'W2': self.W2})
        texture_feat2 = compact_bilinear_pooling(activations2[self.texture_layer], {'W1': self.W1, 'W2': self.W2})
        texture_loss = F.mean_squared_error(texture_feat1, texture_feat2)

        self.loss = loss1 + loss2 + texture_loss
        self.accuracy = F.accuracy(h, t)
        chainer.report({'loss': self.loss, 'acc': self.accuracy}, self)
        return self.loss
