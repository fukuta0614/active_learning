# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os, sys, time

import chainer
from chainer import cuda, serializers, iterators
import chainer.functions as F
import chainer.links as L
import datetime
import logger
import copy
from active_mnist import active_annotation, initialize_labeled_dataset, query_dataset, Mnist
from models import CNN, CNN_keras

import debugger


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
                    self.whole_count,
                    learning_epoch,
                    self.args.epoch_interval,
                    str(datetime.timedelta(seconds=duration)).split('.')[0],
                    throughput
                )
            )

        self.previous_time = time.time()

    def reset(self):
        self.count = 0


def progress_print(line):
    sys.stderr.write('\r\033[K' + line)
    sys.stderr.flush()


def evaluate(model, test_iter, device):
    test_loss, test_acc = 0, 0
    it = copy.copy(test_iter)
    count = 0
    for batch in it:
        count += len(batch)
        x, t = chainer.dataset.concat_examples(batch, device)
        model(x, t)
        test_loss += model.loss.data * len(batch)
        test_acc += model.accuracy.data * len(batch)

    # log
    logger.plot('test_loss', cuda.to_cpu(test_loss) / count)
    logger.plot('test_accuracy', cuda.to_cpu(test_acc) / count)


def validate(model, val_iter, device):
    val_loss = 0
    it = copy.copy(val_iter)
    count = 0
    for batch in it:
        count += len(batch)
        x, t = chainer.dataset.concat_examples(batch, device)
        model(x, t)
        val_loss += model.loss.data * len(batch)
    # log
    val_loss = cuda.to_cpu(val_loss) / count
    return val_loss


def main():
    parser = argparse.ArgumentParser(description='gpat train')
    parser.add_argument("out")
    parser.add_argument('--log_dir', default='runs_active_new')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--init', '-i', default=None,
                        help='Initialize the model from given file')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='learning minibatch size')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--use_dropout', action='store_true', default=False)

    parser.add_argument('--epoch_interval', default=50, type=int)

    parser.add_argument('--active_sample_size', type=int, default=10)
    parser.add_argument('--no-every_init', dest='every_init', action='store_false', default=True)

    parser.add_argument('--random_sample', action='store_true', default=False)
    parser.add_argument('--fixed_ratio', action='store_true', default=False)
    parser.add_argument('--label_init', choices=['random', 'clustering'], default='clustering')
    parser.add_argument('--init_size', default=20, type=int)

    parser.add_argument('--uncertain', action='store_true', default=False)
    parser.add_argument('--uncertain_with_dropout', action='store_true', default=False)
    parser.add_argument('--uncertain_strategy', choices=['entropy', 'least_confident', 'margin'], default='margin')

    parser.add_argument('--clustering', action='store_true', default=False)
    parser.add_argument('--kmeans_cache', default='initial_clustering_result.pkl')

    parser.add_argument('--query_by_committee', action='store_true', default=False)
    parser.add_argument('--qbc_strategy', choices=['vote', 'average_kl'], default='average_kl')
    parser.add_argument('--committee_size', default=10, type=int)

    parser.add_argument('--aug_in_inference', action='store_true', default=False)

    args = parser.parse_args()

    device = args.gpu

    # os.environ['PATH'] += ':/usr/local/cuda/bin'

    # log directory
    logger.init(args)

    # load data
    train, test = chainer.datasets.get_mnist()
    test = Mnist(*test._datasets, aug=False)
    test_iter = iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    val = np.load('val.pkl')
    val = Mnist(*val, aug=False)
    val_iter = iterators.SerialIterator(val, args.batch_size, repeat=False, shuffle=False)

    labeled_data, unlabeled_data = initialize_labeled_dataset(args, train._datasets)
    print('now {} labeled samples, {} unlabeled'.format(len(labeled_data[0]), len(unlabeled_data[0])))

    # start training
    reporter = ProgresssReporter(args)
    # total_loss_history = []
    # loss_history = [0]
    for iteration in range(100):

        # model construct
        model = L.Classifier(CNN(10))
        if iteration == 0:
            chainer.serializers.save_npz(os.path.join(logger.out_dir, 'models', 'cnn.model'), model)
        else:
            chainer.serializers.load_npz(os.path.join(logger.out_dir, 'models', 'cnn.model'), model)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        # use gpu
        cuda.get_device_from_id(device).use()
        model.to_gpu()

        labeled_iter = iterators.SerialIterator(Mnist(*labeled_data, aug=True), args.batch_size)
        unlabeled_iter = iterators.SerialIterator(Mnist(*unlabeled_data, aug=args.aug_in_inference), args.batch_size,
                                                  repeat=False, shuffle=False)

        # train phase
        count = 0
        train_loss = 0
        train_acc = 0
        epoch_interval = args.epoch_interval
        anneal_epoch = int(epoch_interval * 0.8)
        min_val_loss = float('inf')
        tol = 0
        anneal_time = 0
        while labeled_iter.epoch < epoch_interval:
            # train with labeled dataset
            batch = labeled_iter.next()
            x, t = chainer.dataset.concat_examples(batch, device=device)
            optimizer.update(model, x, t)
            reporter(labeled_iter.epoch)

            if labeled_iter.is_new_epoch and labeled_iter.epoch == anneal_epoch:
                optimizer.alpha *= 0.1

            if labeled_iter.epoch > args.epoch_interval - 5:
                count += len(batch)
                train_loss += model.loss.data * len(batch)
                train_acc += model.accuracy.data * len(batch)

            # if labeled_iter.is_new_epoch and len(labeled_data[0]) > 100:
            #     val_loss = validate(model, val_iter, device)
            #     print(min_val_loss, val_loss)
            #     if val_loss > min_val_loss:
            #         tol += 1
            #     else:
            #         min_val_loss = val_loss

        # train_loss_tmp = cuda.to_cpu(train_loss) / len(labeled_iter.dataset)
        #     loss_history.append(train_loss_tmp - np.sum(loss_history))

        reporter.reset()

        logger.plot('train_loss', float(train_loss) / count)
        logger.plot('train_accuracy', float(train_acc) / count)

        # test
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            evaluate(model, test_iter, device)

        # logger
        logger.flush()

        if len(labeled_data[0]) >= 1000:
            print('done')
            exit()

        # if len(labeled_data[0]) >= 100:
        #     with open(os.path.join(logger.out_dir,  'labeled_data.pkl'), 'wb') as f:
        #         import pickle
        #         pickle.dump(labeled_data, f)
        #     chainer.serializers.save_npz(os.path.join(logger.out_dir, 'cnn.model'), model)
        #     print('done')
        #     exit()

        # total_loss_history.append((len(labeled_iter.dataset), loss_history))
        # loss_history = []

        preds = np.zeros((args.committee_size, len(unlabeled_iter.dataset), 10))
        # feat = np.zeros((len(unlabeled_iter.dataset), 784))
        if args.random_sample:
            query_indices = np.random.permutation(len(unlabeled_data[0]))[:args.active_sample_size]
        else:
            loop_num = args.committee_size
            for loop in range(loop_num):
                count = 0
                for batch in unlabeled_iter:
                    x, t = chainer.dataset.concat_examples(batch, device=device)
                    with chainer.no_backprop_mode():
                        y = F.softmax(model.predictor(x))
                    preds[loop, count:count + len(batch)] = cuda.to_cpu(y.data)
                    count += len(batch)
                    # if loop == 0:
                    #     feat[i * batch_size: (i + 1) * batch_size] = cuda.to_cpu(x)
                unlabeled_iter.reset()
            query_indices = active_annotation(preds, feat=unlabeled_data[0], opt=args)

        # active sampling
        print('\nactive sampling: ', end='')

        if iteration % 10 == 0:
            logger.save(model, unlabeled_data[0][query_indices], unlabeled_data[1][query_indices])

        labeled_data, unlabeled_data = query_dataset(labeled_data, unlabeled_data, query_indices)
        print('now {} labeled samples, {} unlabeled'.format(len(labeled_data[0]), len(unlabeled_data[0])))


# with open('{}_loss_history.pkl'.format(args.out), 'wb') as f:
#     import pickle
#     pickle.dump(total_loss_history, f)


if __name__ == '__main__':
    main()
