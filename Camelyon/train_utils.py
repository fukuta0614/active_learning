import os
import sys
import time
import datetime
import random
import numpy as np
import copy

import chainer
import chainer.functions as F
from chainer import cuda
from chainer.dataset import convert
import logger

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))


def progress_report(count, start_time, batchsize, whole_sample):
    duration = time.time() - start_time
    throughput = count * batchsize / duration
    sys.stderr.write(
        '\r{} updates ({} / {} samples) time: {} ({:.2f} samples/sec)'.format(
            count, count * batchsize, whole_sample, str(datetime.timedelta(seconds=duration)).split('.')[0], throughput
        )
    )


class ProgresssReporter(object):
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.previous_time = time.time()
        self.count = 0
        self.whole_count = 0

    def __call__(self, learning_epoch=None):
        self.count += 1
        self.whole_count += 1

        duration = time.time() - self.start_time
        throughput = self.args.batch_size / (time.time() - self.previous_time)
        if self.args.epoch_interval == 0:
            sys.stderr.write(
                '\r{} / {} updates ({} / {} iterations) time: {} ({:.2f} samples/sec)'.format(
                    self.whole_count,
                    self.args.iterations,
                    self.count,
                    self.args.interval,
                    str(datetime.timedelta(seconds=duration)).split('.')[0],
                    throughput
                )
            )
        else:
            sys.stderr.write(
                '\r{} updates ({} / {} epoch) time: {} ({:.2f} samples/sec)'.format(
                    self.count,
                    learning_epoch,
                    self.args.epoch_interval,
                    str(datetime.timedelta(seconds=duration)).split('.')[0],
                    throughput
                )
            )

        self.previous_time = time.time()

    def reset(self):
        self.count = 0


def random_crop(images):
    crop_size = np.random.choice([320, 384, 448, 500])
    _, _, h, w = images.shape
    # Randomly crop a region and flip the image
    top = random.randint(0, h - crop_size - 1)
    left = random.randint(0, w - crop_size - 1)
    if random.randint(0, 1):
        images = images[:, :, :, ::-1]
    bottom = top + crop_size
    right = left + crop_size
    return images[:, :, top:bottom, left:right]


def make_optimizer(model, opt, lr=1e-4):
    if opt == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=lr)
    elif opt == 'momentum':
        optimizer = chainer.optimizers.MomentumSGD(lr=lr)
    elif opt == 'rmsprop':
        optimizer = chainer.optimizers.RMSprop(lr=lr)
    else:
        raise ValueError('invalid optimizer name')

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00005))

    return optimizer


def evaluate(model, diag_iter, device):
    pred_all = []
    ans_all = []

    for diag in diag_iter:
        test_loss = 0
        it = copy.copy(diag_iter[diag])
        pred = np.zeros(len(it.dataset))
        ans = np.zeros(len(it.dataset))
        count = 0
        for batch in it:
            x, t = convert.concat_examples(batch, device)
            model(x, t)
            pred[count: count + len(batch)] = cuda.to_cpu(F.softmax(model.y).data[:, 1])
            ans[count: count + len(batch)] = cuda.to_cpu(t)
            count += len(batch)
            test_loss += model.loss.data * len(batch)

        ans_all.extend(ans)
        pred_all.extend(pred)
        pred, ans = np.array(pred), np.array(ans)
        pred_points = [1 * (pred > p) for p in [0.35, 0.5, 0.75, 0.9]]

        accuracy_list = [np.mean(pred_point == ans) for pred_point in pred_points]

        # save mistake samples
        # mistake = np.array(diag_iter[diag].dataset.base)[ans != pred]
        # np.save(os.path.join(logger.out_dir, "mistakes", '{}_mistake_{:.3f}'.format(diag, accuracy_list[1])), mistake)

        # log
        logger.plot(diag + '_loss', cuda.to_cpu(test_loss) / count)
        logger.plot(diag + '_accuracy', accuracy_list)

    logger.plot_score(pred_all, ans_all, model)


def evaluate_ex(model, test_iter, device):
    test_loss = 0
    test_accuracy = 0
    it = copy.copy(test_iter)
    pred = np.zeros(len(it.dataset))
    ans = np.zeros(len(it.dataset))
    count = 0
    for batch in it:
        x, t = convert.concat_examples(batch, device)
        model(x, t)
        pred[count: count + len(batch)] = cuda.to_cpu(F.softmax(model.y).data[:, 1])
        ans[count: count + len(batch)] = cuda.to_cpu(t)
        count += len(batch)
        test_loss += model.loss.data * len(batch)
        test_accuracy += model.accuracy.data * len(batch)

    logger.plot('test_loss', cuda.to_cpu(test_loss) / count)
    logger.plot('test_accuracy', cuda.to_cpu(test_accuracy) / count)

    normal_accuracies = []
    tumor_accuracies = []
    test_accuracies = []

    for th in [0.35, 0.5, 0.75, 0.9]:
        test_accuracies.append(np.mean(1 * (pred > th) == ans))
        normal_accuracies.append(np.mean(pred[ans == 0] < th))
        tumor_accuracies.append(np.mean(pred[ans == 1] > th))

    # save mistake samples
    # mistake = np.array(diag_iter[diag].dataset.base)[ans != pred]
    # np.save(os.path.join(logger.out_dir, "mistakes", '{}_mistake_{:.3f}'.format(diag, accuracy_list[1])), mistake)

    # log
    logger.plot('normal_accuracy', normal_accuracies)
    logger.plot('tumor_accuracy', tumor_accuracies)
    logger.plot('test_accuracies', test_accuracies)

    logger.plot_score(pred, ans, model)
