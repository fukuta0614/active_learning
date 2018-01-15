# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os, sys, time

import chainer
from chainer import cuda, serializers, iterators
import chainer.functions as F
import chainer.links as L

from camelyon_utils import CamelyonDatasetFromTif, CamelyonDatasetEx, dataset_path
from train_utils import make_optimizer, ProgresssReporter, random_crop, evaluate_ex
from active_utils import active_annotation, initialize_labeled_dataset, query_dataset
import logger

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
from models import TrainableCNN, BilinearCNN
import debugger


def main():
    parser = argparse.ArgumentParser(description='gpat train')
    parser.add_argument("out")
    parser.add_argument('--resume', default=None)
    parser.add_argument('--log_dir', default='runs_active')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--iterations', default=10 ** 5, type=int,
                        help='number of iterations to learn')
    parser.add_argument('--interval', default=100, type=int,
                        help='number of iterations to evaluate')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='learning minibatch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loaderjob', type=int, default=8)
    parser.add_argument('--hed', dest='hed', action='store_true', default=False)
    parser.add_argument('--from_tiff', dest='from_tiff', action='store_true', default=False)
    parser.add_argument('--no-texture', dest='texture', action='store_false', default=True)
    parser.add_argument('--cbp', dest='cbp', action='store_true', default=False)
    parser.add_argument('--no-color_aug', dest='color_aug', action='store_false', default=True)
    parser.add_argument('--model_test', default='', type=str)

    parser.add_argument('--arch', default='googlenet',
                        choices=['texturecnn', 'resnet', 'googlenet', 'vgg', 'alex', 'trained', 'resume'])
    parser.add_argument('--opt', default='adam', choices=['adam', 'momentum'])
    parser.add_argument('--train_path', default='train_extracted_dataset.pkl')
    parser.add_argument('--test_path', default='test_extracted_dataset.pkl')

    parser.add_argument('--epoch_interval', default=20, type=int)

    parser.add_argument('--active_sample_size', type=int, default=100)
    parser.add_argument('--no-every_init', dest='every_init', action='store_false', default=True)

    parser.add_argument('--random_sample', action='store_true', default=False)
    parser.add_argument('--fixed_ratio', action='store_true', default=False)
    parser.add_argument('--label_init', choices=['random', 'clustering'], default='clustering')
    parser.add_argument('--init_size', default=100, type=int)

    parser.add_argument('--uncertain', action='store_true', default=False)
    parser.add_argument('--uncertain_with_dropout', action='store_true', default=False)
    parser.add_argument('--uncertain_strategy', choices=['entropy', 'least_confident', 'margin'], default='margin')

    parser.add_argument('--clustering', action='store_true', default=False)
    parser.add_argument('--kmeans_cache', default='initial_clustering_result.pkl')
    parser.add_argument('--initial_label_cache', default='initial_label_cache.npy')

    parser.add_argument('--query_by_committee', action='store_true', default=False)
    parser.add_argument('--qbc_strategy', choices=['vote', 'average_kl'], default='average_kl')
    parser.add_argument('--committee_size', default=10, type=int)

    parser.add_argument('--aug_in_inference', action='store_true', default=False)

    args = parser.parse_args()

    device = args.gpu

    # log directory
    logger.init(args)

    # load data
    train_dataset = np.load(os.path.join(dataset_path, args.train_path))
    test_dataset = np.load(os.path.join(dataset_path, args.test_path))
    num_class = 2
    image_size = 256
    crop_size = 224

    preprocess_type = args.arch if not args.hed else 'hed'
    perm = np.random.permutation(len(test_dataset))[:10000]
    test_dataset = [test_dataset[idx] for idx in perm]
    test = CamelyonDatasetEx(test_dataset, original_size=image_size, crop_size=crop_size,
                             aug=False, color_aug=False, preprocess_type=preprocess_type)
    test_iter = iterators.MultiprocessIterator(test, args.batch_size, repeat=False, shuffle=False)

    cbp_feat = np.load('train_cbp512_feat.npy')
    labeled_data, unlabeled_data, feat = initialize_labeled_dataset(args, train_dataset, cbp_feat)
    print('now {} labeled samples, {} unlabeled'.format(len(labeled_data), len(unlabeled_data)))

    # start training
    reporter = ProgresssReporter(args)
    for iteration in range(100):

        # model construct
        if args.texture:
            model = BilinearCNN(base_cnn=args.arch, pretrained_model='auto', num_class=num_class,
                                texture_layer=None, cbp=args.cbp, cbp_size=4096)
        else:
            model = TrainableCNN(base_cnn=args.arch, pretrained_model='auto', num_class=num_class)

        # set optimizer
        optimizer = make_optimizer(model, args.opt, args.lr)

        # use gpu
        cuda.get_device_from_id(device).use()
        model.to_gpu()

        labeled_dataset = CamelyonDatasetEx(labeled_data, original_size=image_size, crop_size=crop_size,
                                            aug=True, color_aug=True, preprocess_type=preprocess_type)
        labeled_iter = iterators.MultiprocessIterator(labeled_dataset, args.batch_size)

        # train phase
        count = 0
        train_loss = 0
        train_acc = 0
        epoch_interval = args.epoch_interval if len(labeled_data[0]) < 10000 else args.epoch_interval * 2
        anneal_epoch = int(epoch_interval * 0.8)
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

                # if labeled_iter.is_new_epoch:
                #     train_loss_tmp = cuda.to_cpu(train_loss) / len(labeled_iter.dataset)
                #     loss_history.append(train_loss_tmp - np.sum(loss_history))

        reporter.reset()

        logger.plot('train_loss', cuda.to_cpu(train_loss) / count)
        logger.plot('train_accuracy', cuda.to_cpu(train_acc) / count)

        # test
        print('\ntest')
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            evaluate_ex(model, test_iter, device)

        # logger
        logger.flush()

        if len(labeled_data[0]) >= 10000:
            print('done')
            exit()

        tmp_indices = np.random.permutation(len(unlabeled_data))[:10000]
        tmp_unlabeled_data = [unlabeled_data[idx] for idx in tmp_indices]
        tmp_cbp_feat = cbp_feat[tmp_indices]

        unlabeled_dataset = CamelyonDatasetEx(tmp_unlabeled_data, original_size=image_size, crop_size=crop_size,
                                              aug=args.aug_in_inference, color_aug=args.aug_in_inference,
                                              preprocess_type=preprocess_type)
        unlabeled_iter = iterators.MultiprocessIterator(unlabeled_dataset, args.batch_size,
                                                        repeat=False, shuffle=False)

        preds = np.zeros((args.committee_size, len(tmp_unlabeled_data), 2))
        # feat = np.zeros((len(unlabeled_iter.dataset), 784))
        if args.random_sample:
            tmp_query_indices = np.random.permutation(len(tmp_unlabeled_data))[:args.active_sample_size]
        else:
            loop_num = args.committee_size
            for loop in range(loop_num):
                count = 0
                for batch in unlabeled_iter:
                    x, t = chainer.dataset.concat_examples(batch, device=device)
                    with chainer.no_backprop_mode():
                        y = F.softmax(model.forward(x))
                    preds[loop, count:count + len(batch)] = cuda.to_cpu(y.data)
                    count += len(batch)
                    # if loop == 0:
                    #     feat[i * batch_size: (i + 1) * batch_size] = cuda.to_cpu(x)
                unlabeled_iter.reset()
            tmp_query_indices = active_annotation(preds, tmp_cbp_feat, opt=args)

        # active sampling
        print('active sampling: ', end='')

        if iteration % 10 == 0:
            logger.save(model, [tmp_unlabeled_data[idx] for idx in tmp_query_indices])

        query_indices = tmp_indices[tmp_query_indices]
        labeled_data, unlabeled_data, cbp_feat = query_dataset(labeled_data, unlabeled_data, cbp_feat, query_indices)
        print('now {} labeled samples, {} unlabeled'.format(len(labeled_data), len(unlabeled_data)))


if __name__ == '__main__':
    main()
