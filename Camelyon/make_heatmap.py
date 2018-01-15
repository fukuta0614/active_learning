import argparse
import os
import sys
import numpy as np
import pandas as pd
import openslide
import chainer
import chainer.functions as F
from chainer import cuda, serializers, iterators

from train_utils import MODEL_PATH

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
from models import resnet50, googlenet, vgg, cnn
from PIA.camelyon import CamelyonDataset, dataset_path, base_dir_17

import time

archs = {
    'texturecnn': cnn.TextureCNN(density=2, channel=3),
    'googlenet': googlenet.GoogLeNet,
    'resnet': resnet50.ResNet,
    'vgg': vgg.VGG16
}

init_path = {
    'googlenet': 'googlenet.npz',
    'resnet': 'ResNet-50-model.npz',
    'vgg': 'VGG_ILSVRC_16_layers.npz'
}


def main():
    parser = argparse.ArgumentParser(description='gpat train ')
    parser.add_argument('src')
    parser.add_argument("out")
    parser.add_argument('--model', default=None)
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batch_size', '-b', type=int, default=256,
                        help='learning minibatch size')
    parser.add_argument('--target', '-t', type=int, choices=[16, 17], default=17)
    parser.add_argument('--no-texture', dest='texture', action='store_false', default=True)
    parser.add_argument('--no-normalize', dest='normalize', action='store_false', default=True)
    parser.add_argument('--cbp', dest='cbp', action='store_true', default=False)
    parser.add_argument('--hed', dest='hed', action='store_true', default=False)
    parser.add_argument('--arch', default='googlenet',
                        choices=['texturecnn', 'resnet', 'googlenet', 'vgg', 'alex', 'trained'])
    args, _ = parser.parse_known_args()

    device = args.gpu

    model = archs[args.arch](texture=args.texture, cbp=args.cbp, normalize=args.normalize)
    model.load_pretrained(os.path.join(MODEL_PATH, init_path[args.arch]), num_class=2)

    model_path = os.path.join('runs_16', args.model, 'models',
                              sorted(os.listdir(os.path.join('runs_16', args.model, 'models')))[-1])
    print(model_path)
    chainer.serializers.load_npz(model_path, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(device).use()
        model.to_gpu()

    if args.target == 16:
        # base_dir = '/media/8T-HDD-2/CAMELYON16/Testset/'
        base_dir = '/data/unagi0/fukuta/CAMELYON16/'

        csv = pd.read_csv(os.path.join(base_dir, 'Testset/Ground_Truth', "GT.csv"))
        ans_dict = {x[0]: x[3] for x in csv.values}

        valid_patch_dir = os.path.join(base_dir, 'valid_patchs', 'valid_patch_256')
        valid = sorted([os.path.join(valid_patch_dir, x) for x in os.listdir(valid_patch_dir) if not "Test" in x])

    elif args.target == 17:
        # base_dir = '/media/8T-HDD-2/CAMELYON16/Testset/'
        # base_dir = '/data/unagi0/fukuta/CAMELYON17/'
        # valid = np.load(os.path.join(base_dir, 'training/test_17.npy'))

        # src_dir = os.path.join(base_dir, 'Valid_Patch_New_256')
        # src_dir = os.path.join(base_dir, 'valid_patch_512_overlap')
        # src_dir = os.path.join(base_dir, 'valid_patch_512')

        src_dir = os.path.join(base_dir_17, 'valid_patchs', args.src)
        valid = sorted([os.path.join(src_dir, patient, path)
                        for patient in os.listdir(src_dir)
                        for path in os.listdir(os.path.join(src_dir, patient))
                        if 'mask' not in path])
    else:
        raise ValueError('invalid target')

    if '512' in src_dir:
        image_size = 512
        crop_size = 512
    else:
        image_size = 256
        crop_size = 224

    # heatmap_dir = os.path.join(base_dir, 'heatmap_512_texture_overlap')
    heatmap_dir = os.path.join(base_dir_17, 'heatmaps', args.out)
    os.makedirs(heatmap_dir, exist_ok=True)

    def make_heatmap(path):
        s = time.time()

        slide = path.split('/')[-1].split('.')[0]
        patches = np.load(path)
        if len(patches) == 1:
            print(slide, 'not yet')
            return

        print(slide)
        ans_dir = os.path.join(heatmap_dir, slide)
        try:
            os.makedirs(ans_dir)
        except OSError:
            return

        # print(ans_dict[slide], end=' - ')
        data = [slide + '_' + p for p in patches]

        preprocess_type = args.arch if not args.hed else 'hed'
        val = CamelyonDataset(data, original_size=image_size, crop_size=crop_size, aug=False, color_aug=False,
                              num_class=2, from_tif=True, preprocess_type=preprocess_type)
        val_iter = iterators.MultiprocessIterator(val, args.batch_size, repeat=False, shuffle=False)

        pred = []
        ans = []
        count = 0
        N = len(patches)
        for batch in val_iter:
            count += len(batch)
            print('\r{}/{} ({:.2f} samples/sec)'.format(count, N, count / (time.time() - s)))
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                x = chainer.Variable(model.xp.asarray([b[0] for b in batch], 'float32'))
                # res = list(cuda.to_cpu(F.softmax(model.forward(x)).data))
                res = list(cuda.to_cpu(F.softmax(model.forward(x)).data)[:, 1])
            y = [b[1] for b in batch]
            ans.extend(y)
            pred.extend(res)

        height, width = [int(x) for x in patches[0].split('_')[1:4:2]]
        # img = np.zeros((height, width, 3))
        img = np.zeros((height, width))
        img_tumor = np.zeros((height, width)) == 1

        for idx, path in enumerate(patches):
            i, j = [int(x) for x in path.split('.')[0].split('_')[:3:2]]
            img[i][j] = pred[idx]
            if path.split('_')[-1] == 'tumor':
                img_tumor[i][j] = True

        np.save(os.path.join(ans_dir, slide + '_pred'), img)
        np.save(os.path.join(ans_dir, slide + '_tumor'), img_tumor)

    print(image_size)
    for x in valid:
        make_heatmap(x)


if __name__ == '__main__':
    main()
