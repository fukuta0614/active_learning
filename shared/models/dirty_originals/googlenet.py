import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import sys, os

shared = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(shared)
from ..functions import power_normalize, global_average_pooling_2d, bilinear_pooling


class GoogLeNet(chainer.Chain):
    insize = 224

    def __init__(self, num_class=1000, texture=False, cbp=False, normalize=True, mil=False):
        super(GoogLeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, stride=2, pad=3)
            self.conv2_reduce = L.Convolution2D(None, 64, 1)
            self.conv2 = L.Convolution2D(None, 192, 3, stride=1, pad=1)
            self.inc3a = L.Inception(None, 64, 96, 128, 16, 32, 32)
            self.inc3b = L.Inception(None, 128, 128, 192, 32, 96, 64)
            self.inc4a = L.Inception(None, 192, 96, 208, 16, 48, 64)
            self.inc4b = L.Inception(None, 160, 112, 224, 24, 64, 64)
            self.inc4c = L.Inception(None, 128, 128, 256, 24, 64, 64)
            self.inc4d = L.Inception(None, 112, 144, 288, 32, 64, 64)

            if not texture:
                self.inc4e = L.Inception(None, 256, 160, 320, 32, 128, 128)
                self.inc5a = L.Inception(None, 256, 160, 320, 32, 128, 128)
                self.inc5b = L.Inception(None, 384, 192, 384, 48, 128, 128)

                self.loss1_conv = L.Convolution2D(None, 128, 1)
                self.loss1_fc1 = L.Linear(None, 1024)
                self.loss1_fc2 = L.Linear(None, num_class)

                self.loss2_conv = L.Convolution2D(None, 128, 1)
                self.loss2_fc1 = L.Linear(None, 1024)
                self.loss2_fc2 = L.Linear(None, num_class)

            self.loss3_fc = L.Linear(None, num_class)

        self.texture = texture
        self.cbp = cbp
        self.normalize = normalize
        self.mil = mil

    def forward(self, x, every_output=False, small_cbp=False):
        """
        inc4e  :  (1, 832, 14, 14)
        pooling : (1, 832, 7, 7)
        inc5a : (1, 832, 7, 7)
        inc5b : (1, 1024, 7, 7)
        last : (1, 1024, 1, 1)
        :param x:
        :param every_output:
        :return:
        """
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)

        if every_output:
            l = F.average_pooling_2d(h, 5, stride=3)
            l = F.relu(self.loss1_conv(l))
            l = F.relu(self.loss1_fc1(l))
            l1 = self.loss1_fc2(l)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        if self.texture:
            if small_cbp:

                _h = F.convolution_2d(h, self.W1_256) * F.convolution_2d(h, self.W2_256)
                _h = global_average_pooling_2d(_h)
                if self.normalize:
                    _h = power_normalize(_h)
                    _h = F.normalize(_h)
                self.cbp_feat = _h

            if self.cbp:
                h = F.convolution_2d(h, self.W1) * F.convolution_2d(h, self.W2)
                h = global_average_pooling_2d(h)
                if self.normalize:
                    h = power_normalize(h)
                    h = F.normalize(h)
                h = self.loss3_fc(F.dropout(h, 0.25))
                return h
            else:
                b, ch, height, width = h.data.shape
                h = F.reshape(h, (b, ch, width * height))
                h = F.batch_matmul(h, h, transb=True) / self.xp.float32(width * height)

                # h = F.reshape(h, (b, ch * ch)) #TODO(変更点）

                if self.normalize:
                    h = power_normalize(h)
                    h = F.normalize(h)
                h = self.loss3_fc(F.dropout(h, 0.25))
                return h

        else:
            if every_output:
                l = F.average_pooling_2d(h, 5, stride=3)
                l = F.relu(self.loss2_conv(l))
                l = F.relu(self.loss2_fc1(l))
                l2 = self.loss2_fc2(l)

            h = self.inc4e(h)
            h = F.max_pooling_2d(h, 3, stride=2)
            h = self.inc5a(h)
            h = self.inc5b(h)
            h = F.average_pooling_2d(h, 7, stride=1)
            if self.mil:
                h = F.max(h, axis=0, keepdims=True)
            h = self.loss3_fc(F.dropout(h, 0.4))

            if every_output:
                return l1, l2, h
            else:
                return h

    def load_pretrained(self, pretrained_path, num_class):
        # self._persistent.discard('W1')
        # self._persistent.discard('W2')
        # self.loss3_fc = L.Linear(None, 1000)
        chainer.serializers.load_npz(pretrained_path, self)
        self.convert_to_finetune_model(num_class)

    def convert_to_finetune_model(self, num_class):
        if self.cbp:
            randweight = np.load(os.path.join(shared, 'cbp/randweight_528_to_4096.npz'))
            self.add_persistent('W1', randweight['W1'])
            self.add_persistent('W2', randweight['W2'])

            randweight_256 = np.load(os.path.join(shared, 'cbp/randweight_528_to_256.npz'))
            self.add_persistent('W1_256', randweight_256['W1'])
            self.add_persistent('W2_256', randweight_256['W2'])

        self.loss3_fc = L.Linear(None, num_class)

    def __call__(self, x, t, every_loss=False):

        if every_loss:
            l1, l2, l3 = self.forward(x, every_output=True)
            loss1 = F.softmax_cross_entropy(l1, t)
            loss2 = F.softmax_cross_entropy(l2, t)
            loss3 = F.softmax_cross_entropy(l3, t)
            self.loss = 0.3 * (loss1 + loss2) + loss3
            self.accuracy = F.accuracy(l3, t)
            return self.loss
        else:
            self.y = self.forward(x, every_output=False)
            self.loss = F.softmax_cross_entropy(self.y, t)
            self.accuracy = F.accuracy(self.y, t)
            return self.loss