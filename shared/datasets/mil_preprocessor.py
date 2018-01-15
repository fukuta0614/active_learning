import os
from PIL import Image
import cv2
from skimage.color import rgb2hed
import numpy as np
import random


class MILPreprocessor(object):
    def __init__(self, pairs, original_size=256, crop_size=224, hed=True, random=True, mean=0):
        self.base = pairs
        self.image_size = original_size
        self.crop_size = crop_size
        self.hed = hed
        self.random = random
        self.mean = mean

    def __len__(self):
        return len(self.base)

    def get_bag(self, i, batch_size):

        image_dir, label = self.base[i]
        bag = []
        try:
            image_path = np.random.choice(os.listdir(image_dir), size=batch_size, replace=False)
        except ValueError:
            image_path = os.listdir(image_dir)

        for path in image_path:
            bag.append(self.get_example(os.path.join(image_dir, path)))

        return bag, label

    def get_example(self, path):

        # load data
        with Image.open(path) as f:
            image = np.asarray(f)

        # rotation
        rotation_matrix = cv2.getRotationMatrix2D((self.image_size / 2, self.image_size / 2), 90 * random.randint(0, 3),
                                                  1.0)
        image = cv2.warpAffine(image, rotation_matrix, (self.image_size, self.image_size), flags=cv2.INTER_CUBIC)

        # rgb2hed
        if self.hed:
            image = rgb2hed(image)
        else:
            image = image[::-1, :, :] - self.mean

        # transpose
        image = image.transpose(2, 0, 1).astype('float32')

        # crop
        crop_size = self.crop_size
        _, h, w = image.shape
        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]

        # preprocess
        # image /= 255

        return image
