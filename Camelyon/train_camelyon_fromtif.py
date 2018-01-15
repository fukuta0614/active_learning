# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os, sys, time

import chainer
from chainer import cuda, serializers, iterators
from chainer.training import updaters

from camelyon_utils import CamelyonDatasetFromTif, CamelyonDatasetEx, dataset_path
from train_utils import make_optimizer, progress_report, evaluate_ex

import logger

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
from models import TrainableCNN, BilinearCNN
import debugger


def main():
    parser = argparse.ArgumentParser(description='gpat train ')
    parser.add_argument("out")
    parser.add_argument('--resume', default=None)
    parser.add_argument('--log_dir', default='runs_16')
    parser.add_argument('--gpus', '-g', type=int, nargs="*",
                        default=[0, 1, 2, 3])
    parser.add_argument('--iterations', default=10 ** 5, type=int,
                        help='number of iterations to learn')
    parser.add_argument('--interval', default=1000, type=int,
                        help='number of iterations to evaluate')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='learning minibatch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loaderjob', type=int, default=8)
    parser.add_argument('--hed', dest='hed', action='store_true', default=False)
    # parser.add_argument('--size', '-s', default=96, type=int, choices=[48, 64, 80, 96, 112, 128],
    #                     help='image size')
    parser.add_argument('--no-texture', dest='texture', action='store_false', default=True)
    parser.add_argument('--cbp', dest='cbp', action='store_true', default=False)
    parser.add_argument('--no-color_aug', dest='color_aug', action='store_false', default=True)
    parser.add_argument('--model_test', default='', type=str)
    parser.add_argument('--no-finetune', dest='finetune', action='store_false', default=True)
    parser.add_argument('--arch', default='googlenet',
                        choices=['texturecnn', 'resnet50', 'resnet101', 'googlenet', 'vgg', 'alex', 'trained',
                                 'resume'])
    parser.add_argument('--opt', default='adam', choices=['adam', 'momentum'])
    parser.add_argument('--train_path', default='train_extracted_dataset.pkl')
    parser.add_argument('--test_path', default='test_extracted_dataset.pkl')
    parser.add_argument('--data_size', type=float, default=1.)
    parser.add_argument('--new', action='store_true', default=False)
    args = parser.parse_args()

    devices = tuple(args.gpus)
    # os.environ['PATH'] += ':/usr/local/cuda/bin'

    # log directory
    logger.init(args)

    # load data
    train_dataset = np.load(os.path.join(dataset_path, args.train_path))
    test_dataset = np.load(os.path.join(dataset_path, args.test_path))
    num_class = 2
    image_size = 256
    crop_size = 224

    if 'extracted' in train_dataset:
        perm = np.random.permutation(len(train_dataset))[:int(len(train_dataset) * args.data_size)]
        train_dataset = [train_dataset[p] for p in perm]

    preprocess_type = args.arch if not args.hed else 'hed'
    if 'extracted' in args.train_path:
        train = CamelyonDatasetEx(train_dataset, original_size=image_size, crop_size=crop_size, aug=True,
                                  color_aug=args.color_aug, preprocess_type=preprocess_type)
    else:
        train = CamelyonDatasetFromTif(train_dataset, original_size=image_size, crop_size=crop_size, aug=True,
                                       color_aug=args.color_aug, preprocess_type=preprocess_type)
    if len(devices) > 1:
        train_iter = [
            chainer.iterators.MultiprocessIterator(i, args.batch_size, n_processes=args.loaderjob)
            for i in chainer.datasets.split_dataset_n_random(train, len(devices))]
    else:
        train_iter = iterators.MultiprocessIterator(train, args.batch_size, n_processes=args.loaderjob)

    test = CamelyonDatasetEx(test_dataset, original_size=image_size, crop_size=crop_size, aug=False,
                             color_aug=False, preprocess_type=preprocess_type)
    test_iter = iterators.MultiprocessIterator(test, args.batch_size, repeat=False, shuffle=False)

    # model construct
    if args.texture:
        model = BilinearCNN(base_cnn=args.arch, pretrained_model='auto', num_class=num_class,
                            texture_layer=None, cbp=args.cbp, cbp_size=4096)
    else:
        model = TrainableCNN(base_cnn=args.arch, pretrained_model='auto', num_class=num_class)

    if args.model_test:
        # test
        # model_path = os.path.join('runs_16', args.model_test, 'models',
        #                           sorted(os.listdir(os.path.join('runs_16', args.model_test, 'models')))[-1])
        # print(model_path)
        # chainer.serializers.load_npz(model_path, model)
        cuda.get_device_from_id(devices[0]).use()
        model.to_gpu()
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            evaluate_ex(model, test_iter, devices[0])
        logger.flush()
        exit()

    if args.resume is not None:
        model_path = os.path.join('runs_16', args.resume, 'models',
                                  sorted(os.listdir(os.path.join('runs_16', args.resume, 'models')))[-1])
        print(model_path)
        chainer.serializers.load_npz(model_path, model)

    # set optimizer
    optimizer = make_optimizer(model, args.opt, args.lr)

    if len(devices) > 1:
        updater = updaters.MultiprocessParallelUpdater(train_iter, optimizer, devices=devices)
    else:
        cuda.get_device_from_id(devices[0]).use()
        model.to_gpu()
        # updater
        updater = chainer.training.StandardUpdater(train_iter, optimizer, device=devices[0])

    # start training
    start = time.time()
    train_loss = 0
    train_accuracy = 0
    while updater.iteration < args.iterations:

        # train
        updater.update()
        progress_report(updater.iteration, start, len(devices) * args.batch_size, len(train))
        train_loss += model.loss.data
        train_accuracy += model.accuracy.data

        if updater.iteration % args.interval == 0:
            logger.plot('train_loss', cuda.to_cpu(train_loss) / args.interval)
            logger.plot('train_accuracy', cuda.to_cpu(train_accuracy) / args.interval)
            train_loss = 0
            train_accuracy = 0

            # test
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                evaluate_ex(model, test_iter, devices[0])

            # logger
            logger.flush()

            # save
            serializers.save_npz(os.path.join(logger.out_dir, 'resume'), updater)

            if updater.iteration % 20000 == 0:
                if args.opt == 'adam':
                    optimizer.alpha *= 0.1
                else:
                    optimizer.lr *= 0.1


if __name__ == '__main__':
    main()
