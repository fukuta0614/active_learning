import os
import numpy as np
import chainer
import pickle

import utils as U


class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, images, labels, opt, train=True):
        self.base = chainer.datasets.TupleDataset(images, labels)
        self.opt = opt
        self.train = train
        if opt.dataset == 'cifar10':
            self.mean = np.array([125.3, 123.0, 113.9])
            self.std = np.array([63.0, 62.1, 66.7])
        else:
            self.mean = np.array([129.3, 124.1, 112.4])
            self.std = np.array([68.2, 65.4, 70.4])

        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        return len(self.base)

    def preprocess_setup(self):
        if self.train:
            funcs = [U.normalize(self.mean, self.std),
                     U.horizontal_flip(),
                     U.padding(4),
                     U.random_crop(32),
                     ]
        else:
            funcs = [U.normalize(self.mean, self.std)]

        return funcs

    def preprocess(self, image):
        for f in self.preprocess_funcs:
            image = f(image)

        return image

    def get_example(self, i):
        image, label = self.base[i]
        image = self.preprocess(image).astype(np.float32)
        label = np.array(label, dtype=np.int32)

        return image, label


def setup(opt):
    def unpickle(fn):
        with open(fn, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
        return data

    if opt.dataset == 'cifar10':
        train = [unpickle(os.path.join(opt.data, 'data_batch_{}'.format(i))) for i in range(1, 6)]
        train_images = np.concatenate([d['data'] for d in train]).reshape((-1, 3, 32, 32))
        train_labels = np.concatenate([d['labels'] for d in train])
        val = unpickle(os.path.join(opt.data, 'test_batch'))
        val_images = val['data'].reshape((-1, 3, 32, 32))
        val_labels = val['labels']
    else:
        train = unpickle(os.path.join(opt.data, 'train'))
        train_images = train['data'].reshape(-1, 3, 32, 32)
        train_labels = train['fine_labels']
        val = unpickle(os.path.join(opt.data, 'test'))
        val_images = val['data'].reshape((-1, 3, 32, 32))
        val_labels = val['fine_labels']

    # Iterator setup
    train_data = ImageDataset(train_images, train_labels, opt, train=True)
    val_data = ImageDataset(val_images, val_labels, opt, train=False)
    train_iter = chainer.iterators.MultiprocessIterator(train_data, opt.batchSize, repeat=False)
    val_iter = chainer.iterators.SerialIterator(val_data, opt.batchSize, repeat=False, shuffle=False)

    return train_iter, val_iter
