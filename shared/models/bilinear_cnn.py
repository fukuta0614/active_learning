import os

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import link

from .functions import bilinear_pooling, compact_bilinear_pooling, global_average_pooling_2d
from .cnn_info import CNN_ARCH_INFO

cbp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cbp')


class BilinearCNN(link.Chain):
    def __init__(self, base_cnn, pretrained_model='auto', texture_layer=None, num_class=1000,
                 cbp=False, cbp_size=4096, clustering_cbp=False, clustering_cbp_size=256):
        super(BilinearCNN, self).__init__()
        cf = CNN_ARCH_INFO[base_cnn]
        cnn = cf['cnn'](pretrained_model=pretrained_model)
        self.texture_layer = cf['default_texture_layer'] if texture_layer is None else texture_layer
        feat_size = cf['layer_info'][self.texture_layer]

        self.cbp = cbp
        self.clustering_cbp = clustering_cbp
        self.cbp_feat = None

        with self.init_scope():
            self.cnn = cnn
            self.fc = L.Linear(None, num_class)
            if self.cbp:
                randweight = np.load(
                    os.path.join(cbp_dir, 'randweight_{}_to_{}.npz'.format(feat_size, cbp_size)))
                self.add_persistent('W1', randweight['W1'])
                self.add_persistent('W2', randweight['W2'])
            if self.clustering_cbp:
                randweight = np.load(
                    os.path.join(cbp_dir, 'randweight_{}_to_{}.npz'.format(feat_size, clustering_cbp_size)))
                self.add_persistent('W1_small', randweight['W1'])
                self.add_persistent('W2_small', randweight['W2'])

    def forward(self, x):
        h = self.cnn(x, [self.texture_layer])[self.texture_layer]
        # TODO(for test)
        self.normal_feat = global_average_pooling_2d(h)

        # for clustering (active)
        if self.clustering_cbp:
            self.cbp_feat = compact_bilinear_pooling(h, {'W1': self.W1_small, 'W2': self.W2_small})

        # cbp or original bilinear feature
        if self.cbp:
            h = compact_bilinear_pooling(h, {'W1': self.W1, 'W2': self.W2})
        else:
            h = bilinear_pooling(h)

        # TODO(for test)
        self.feat = h
        h = self.fc(F.dropout(h, 0.4))
        return h

    def __call__(self, x, t):
        self.y = self.forward(x)
        self.loss = F.softmax_cross_entropy(self.y, t)
        self.accuracy = F.accuracy(self.y, t)
        chainer.report({'loss': self.loss, 'acc': self.accuracy}, self)
        return self.loss
