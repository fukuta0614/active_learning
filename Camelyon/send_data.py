# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os, sys, time

# train_data = np.load(os.path.join('dataset', 'train_extracted_dataset.pkl'))
# test_data = np.load(os.path.join('dataset', 'test_extracted_dataset.pkl'))
# from camelyon_utils import extracted_16_dir, extracted_17_dir, extracted_16_512_dir, extracted_17_512_dir


# images = []
# for label, slide_name, patch_info in train_data:
#     _, _, _, _, l, u = patch_info
#     file_name = '{}_{}_{}.png'.format(slide_name, '_'.join([str(_) for _ in patch_info]), label)
#     path = os.path.join(extracted_16_dir, slide_name, file_name)
#     images.append(path)
#
# for label, slide_name, patch_info in test_data:
#     _, _, _, _, l, u = patch_info
#     file_name = '{}_{}_{}.png'.format(slide_name, '_'.join([str(_) for _ in patch_info]), label)
#     path = os.path.join(extracted_16_dir, slide_name, file_name)
#     images.append(path)
#
# np.save('images.npy', images)

images = np.load('/data1/images.npy')

for image in images[:10]:
    print(image)
    slide, image_path = image.split('/')[-2],image.split('/')[-1]
    if not os.path.exists(os.path.join('/data1/Dataset', slide, image_path)):
        command = "scp -r -o ProxyCommand='ssh mil -W %h:%p' " \
                  "maguro:/data/ugui0/fukuta/Camelyon/Dataset16/{0}/{1} /data1/Dataset/{0}/{1}"\
            .format(slide, image_path)
        os.system(command)
