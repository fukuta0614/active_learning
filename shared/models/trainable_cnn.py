import chainer
import chainer.functions as F
import chainer.links as L
from chainer import link
from .cnn_info import CNN_ARCH_INFO


class TrainableCNN(link.Chain):
    def __init__(self, base_cnn, pretrained_model=None, num_class=1000):
        super(TrainableCNN, self).__init__()
        cf = CNN_ARCH_INFO[base_cnn]
        cnn = cf['cnn'](pretrained_model=pretrained_model)
        self.logit = cf['logit']
        if base_cnn == 'googlenet' and pretrained_model is not None:
            self.logit = cf['logit'][:1]

        if num_class != 1000:
            for layer in self.logit:
                setattr(cnn, layer, L.Linear(None, num_class))

        with self.init_scope():
            self.cnn = cnn

    def forward(self, x):
        h = self.cnn(x, layers=self.logit)[self.logit[0]]
        return h

    def __call__(self, x, t):
        activations = self.cnn(x, layers=self.logit)
        self.y = activations[self.logit[0]]
        if len(self.logit) == 3:
            # for googlenet
            loss1, loss2, loss3 = [F.softmax_cross_entropy(activations[layer], t) for layer in self.logit]
            self.loss = 0.3 * (loss1 + loss2) + loss3
        else:
            self.loss = F.softmax_cross_entropy(self.y, t)
        self.accuracy = F.accuracy(self.y, t)
        chainer.report({'loss': self.loss, 'acc': self.accuracy}, self)
        return self.loss
