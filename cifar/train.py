import sys
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import time

from utils import to_hms


class Trainer:
    def __init__(self, model, optimizer, train_iter, val_iter, opt):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.opt = opt
        self.n_batches = (len(train_iter.dataset) - 1) // opt.batchSize + 1
        self.start_time = time.time()

    def train(self, epoch):
        self.optimizer.lr = self.lr_schedule(epoch)
        train_loss = 0
        train_acc = 0
        for i, batch in enumerate(self.train_iter):
            x, t = chainer.dataset.concat_examples(batch, device=self.opt.gpu)
            self.optimizer.zero_grads()
            y = self.model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            loss.backward()
            self.optimizer.update()
            train_loss += float(loss.data) * len(t.data)
            train_acc += float(acc.data) * len(t.data)

            elapsed_time = time.time() - self.start_time
            progress = (self.n_batches * (epoch - 1) + i + 1) * 1.0 / (self.n_batches * self.opt.nEpochs)
            eta = elapsed_time / progress - elapsed_time

            line = '* Epoch: {}/{} ({}/{}) | Train: LR {} | Time: {} (ETA: {})'.format(
                epoch, self.opt.nEpochs, i + 1, self.n_batches,
                self.optimizer.lr, to_hms(elapsed_time), to_hms(eta))
            sys.stderr.write('\r\033[K' + line)
            sys.stderr.flush()

        self.train_iter.reset()
        train_loss /= len(self.train_iter.dataset)
        train_top1 = 100 * (train_acc / len(self.train_iter.dataset))

        return train_loss, train_top1

    def val(self):
        self.model.train = False
        val_acc = 0
        count = 0
        for batch in self.val_iter:
            count += len(batch)
            x_array, t_array = chainer.dataset.concat_examples(batch)
            x = chainer.Variable(cuda.to_gpu(x_array), volatile=True)
            t = chainer.Variable(cuda.to_gpu(t_array), volatile=True)
            y = F.softmax(self.model(x))
            acc = F.accuracy(y, t)
            val_acc += float(acc.data) * len(t.data)
        self.val_iter.reset()
        self.model.train = True
        val_top1 = 100 * (val_acc / len(self.val_iter.dataset))

        return val_top1

    def lr_schedule(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
        decay = sum(epoch > divide_epoch)
        if epoch <= self.opt.warmup:
            decay = 1

        return self.opt.LR * np.power(0.1, decay)
