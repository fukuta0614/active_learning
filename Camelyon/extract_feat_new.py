# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os, sys, time

import chainer
from chainer import cuda, serializers, iterators

from chainer.dataset import convert
from PIL import Image
from camelyon_utils import CamelyonDatasetFromTif, CamelyonDatasetEx, dataset_path

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
from models import TrainableCNN, BilinearCNN
import debugger

from skimage.feature import local_binary_pattern
from skimage.feature import multiblock_lbp
from skimage.transform import integral_image

class LocalBinaryPatterns:
    def __init__(self, radius_list):
        self.radius_list = radius_list

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        feat = []
        for radius in self.radius_list:
            num_points = 8 * radius

            lbp = local_binary_pattern(image, num_points, radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, num_points + 3),
                                     range=(0, num_points + 2))
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            feat.extend(hist)
        return feat


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
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='learning minibatch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loaderjob', type=int, default=8)
    # parser.add_argument('--size', '-s', default=96, type=int, choices=[48, 64, 80, 96, 112, 128],
    #                     help='image size')
    parser.add_argument('--hed', dest='hed', action='store_true', default=False)
    parser.add_argument('--from_tiff', dest='from_tiff', action='store_true', default=False)
    parser.add_argument('--no-texture', dest='texture', action='store_false', default=True)
    parser.add_argument('--cbp', dest='cbp', action='store_true', default=False)
    parser.add_argument('--no-normalize', dest='normalize', action='store_false', default=True)
    parser.add_argument('--color_aug', dest='color_aug', action='store_true', default=False)
    parser.add_argument('--model_test', default='', type=str)
    parser.add_argument('--no-finetune', dest='finetune', action='store_false', default=True)
    parser.add_argument('--arch', default='googlenet',
                        choices=['texturecnn', 'resnet50', 'googlenet', 'vgg', 'alex', 'trained', 'resume'])
    parser.add_argument('--opt', default='adam', choices=['adam', 'momentum'])
    parser.add_argument('--data_size', type=float, default=1)
    # parser.add_argument('--train_path', default='train_0330_additional_new.npy')
    # parser.add_argument('--test_path', default='diag_256_0406.pkl')
    parser.add_argument('--train_path', default='train_extracted_dataset.pkl')
    parser.add_argument('--test_path', default='test_extracted_dataset.pkl')
    parser.add_argument('--new', action='store_true', default=False)
    args = parser.parse_args()

    devices = tuple(args.gpus)
    # os.environ['PATH'] += ':/usr/local/cuda/bin'

    # load data
    train_data = np.load(os.path.join(dataset_path, args.train_path))
    test_data = np.load(os.path.join(dataset_path, args.test_path))

    num_class = 3 if 'three_class' in args.train_path else 2

    if '512' in args.train_path:
        image_size = 512
        crop_size = 384
    else:
        image_size = 256
        crop_size = 224 if not args.arch == 'alex' else 227

    preprocess_type = args.arch if not args.hed else 'hed'
    train = CamelyonDatasetEx(train_data, original_size=image_size, crop_size=crop_size, aug=False,
                              color_aug=False, preprocess_type=preprocess_type)
    test = CamelyonDatasetEx(test_data, original_size=image_size, crop_size=crop_size, aug=False,
                             color_aug=False, preprocess_type=preprocess_type)
    train_iter = iterators.MultiprocessIterator(train, args.batch_size, repeat=False, shuffle=False)
    test_iter = iterators.MultiprocessIterator(test, args.batch_size, repeat=False, shuffle=False)

    # model construct
    model = BilinearCNN(base_cnn=args.arch, pretrained_model='auto', num_class=num_class,
                        texture_layer=None, cbp=args.cbp, cbp_size=512)

    device = devices[0]
    feat_type = 'cbp'

    if feat_type == 'cbp':
        count = 0
        cuda.get_device_from_id(devices[0]).use()
        model.to_gpu()

        feat = np.zeros((len(train_iter.dataset), 512))
        for batch in train_iter:
            x, t = convert.concat_examples(batch, device)
            model(x, t)
            feat[count: count + len(batch)] = cuda.to_cpu(model.feat.data)
            count += len(batch)
            print(count, '/', len(train_iter.dataset))

        np.save('train_cbp512_feat.npy', feat)

        count = 0
        feat = np.zeros((len(test_iter.dataset), 512))
        for batch in test_iter:
            x, t = convert.concat_examples(batch, device)
            model(x, t)
            feat[count: count + len(batch)] = cuda.to_cpu(model.feat.data)
            count += len(batch)
            print(count, '/', len(test_iter.dataset))

        np.save('test_cbp512_feat.npy', feat)

    elif feat_type == 'cnn':
        count = 0
        feat = np.zeros((len(train_iter.dataset), 512))

        cuda.get_device_from_id(devices[0]).use()
        model.to_gpu()

        for batch in tqdm(train_iter, total=len(train_iter.dataset) // args.batch_size):
            x, t = convert.concat_examples(batch, device)
            model(x, t)
            feat[count: count + len(batch)] = cuda.to_cpu(model.normal_feat.data)
            count += len(batch)

        np.save('train_normal_feat.npy', feat)

        for diag in diag_iter:
            import copy
            it = copy.copy(diag_iter[diag])
            feat = np.zeros((len(it.dataset), 512))
            count = 0
            for batch in tqdm(it):
                x, t = convert.concat_examples(batch, device)
                model(x, t)
                feat[count: count + len(batch)] = cuda.to_cpu(model.normal_feat.data)
                count += len(batch)
            np.save('test_' + diag + '_normal_feat.npy', feat)

    else:
        desc = LocalBinaryPatterns([1, 3, 5, 7])

        def extract_lbp(x):
            if 'patient' in x:
                slide_name, (l, u) = '_'.join(x.split('_')[:4]), [int(_) for _ in x.split('_')[8:10]]
                path = os.path.join(extracted_17_dir, slide_name, x + '.png')
            else:
                slide_name, (l, u) = '_'.join(x.split('_')[:2]), [int(_) for _ in x.split('_')[6:8]]
                path = os.path.join(extracted_16_dir, slide_name, x.replace('middle', 'normal') + '.png')

            image = Image.open(path)
            image = image.convert('L')
            feat = desc.describe(np.array(image))

            return feat

        # feat = np.zeros((len(train_data), 136))
        # for i in tqdm(range(len(train_data))):
        #     feat[i] = extract_lbp(train_data[i])
        #
        # np.save('train_lbp_feat.npy', feat)

        for diag in test_data:
            diag_data = test_data[diag]
            feat = np.zeros((len(diag_data), 136))
            for i in tqdm(range(len(diag_data))):
                feat[i] = extract_lbp(diag_data[i])
            np.save('test_' + diag + '_lbp_feat.npy', feat)


if __name__ == '__main__':
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        main()
