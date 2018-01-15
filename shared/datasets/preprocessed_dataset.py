import os
import chainer
from PIL import Image
import cv2
from skimage.color import rgb2hed
import numpy as np
import random
import openslide
from PIL import Image, ImageMath, ImageEnhance, ImageFilter


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, crop_size, aug, color_aug, preprocess_type):
        self.crop_size = crop_size
        self.aug = aug
        self.color_aug = color_aug
        self.preprocess_type = preprocess_type

    def random_crop(self, image: Image):
        org_w, org_h = image.size
        aspect = random.uniform(4 / 5., 5 / 4.)

        if aspect < 1.0:
            w = int(np.random.choice([0.8, 0.9, 1.0]) * org_w)
            h = int(w * aspect)
        else:
            h = int(np.random.choice([0.8, 0.9, 1.0]) * org_h)
            w = int(h / aspect)

        top = random.randint(0, org_h - h)
        left = random.randint(0, org_w - w)

        image = image.crop((left, top, left + w, top + h))
        image = image.resize((self.crop_size, self.crop_size), resample=Image.BICUBIC)

        return image

    def center_crop(self, image: Image):
        org_w, org_h = image.size
        top = (org_h - self.crop_size) // 2
        left = (org_w - self.crop_size) // 2
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        return image

    def preprocess(self, image: Image):
        image = np.asarray(image, dtype=np.float32)

        image = image[:, :, ::-1]
        if 'resnet' in self.preprocess_type:
            image -= np.array([103.063, 115.903, 123.152], dtype=np.float32)
        elif 'vgg' in self.preprocess_type:
            image -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
        elif 'googlenet' in self.preprocess_type:
            image -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
        elif 'hed':
            image = rgb2hed(image)
        else:
            raise ValueError('invalid type')

        image = image.transpose((2, 0, 1))
        return image

    def gaussian_blur(self, image):
        sigma = np.random.uniform(0.0, 2.0)
        img = image.filter(ImageFilter.GaussianBlur(sigma))
        return img

    def color_augment(self, image):
        # 色相いじる
        h, s, v = image.convert("HSV").split()
        _h = ImageMath.eval("(h + {}) % 255".format(np.random.randint(-25, 25)), h=h).convert("L")
        img = Image.merge("HSV", (_h, s, v)).convert("RGB")

        # 彩度を変える
        saturation_converter = ImageEnhance.Color(img)
        img = saturation_converter.enhance(np.random.uniform(0.9, 1.1))

        # コントラストを変える
        contrast_converter = ImageEnhance.Contrast(img)
        img = contrast_converter.enhance(np.random.uniform(0.9, 1.1))

        # 明度を変える
        brightness_converter = ImageEnhance.Brightness(img)
        img = brightness_converter.enhance(np.random.uniform(0.9, 1.1))

        # シャープネスを変える
        sharpness_converter = ImageEnhance.Sharpness(img)
        img = sharpness_converter.enhance(np.random.uniform(0.9, 1.1))

        return img

    def process(self, image):

        image = image.convert("RGB")
        if self.aug:
            # 1. random crop
            image = self.random_crop(image)

            # 2. random rotation (degree, not radian)
            r = np.random.choice([0, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
            image = image.transpose(r) if r > 0 else image
        else:
            image = self.center_crop(image)

        # 3. intensive color augment and blur augment by fukuta
        if self.color_aug:
            image = self.color_augment(image)
            image = self.gaussian_blur(image)

        # 4. preprocess PIL -> numpy array, subtract or divide
        image = self.preprocess(image)

        # 5. Holizontal flip
        if random.randint(0, 1) and self.aug:
            image = image[:, :, ::-1]

        return image


class TCGADataset(chainer.dataset.DatasetMixin):
    def __init__(self, base, size=256,
                 preprocess_type='vgg',
                 root_dir='/data/rama/fukuta/work_space/TCGA_extracted',
                 wsi_dir='/data/rama/DX_DATA'):

        self.base = base
        self.image_size = size
        self.root_dir = root_dir
        self.wsi_dir = wsi_dir
        self.preprocess_type = preprocess_type

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

        # load data
        x = self.base[i]
        path = os.path.join(self.root_dir, x)

        if os.path.exists(path):
            image = np.asarray(Image.open(path))
        else:
            print(path)
            cancer_type, level_and_size, case_id, loc_info = x.split('/')
            wsi_path = os.path.join(self.wsi_dir, cancer_type, case_id + '.svs')
            wsi = openslide.OpenSlide(os.readlink(wsi_path).replace('/data', '/data/rama_hdd'))

            level = int(level_and_size.split('_')[0][-1])
            size = int(level_and_size.split('_')[1])

            l, u = [int(x) * size * int(wsi.level_downsamples[level]) for x in loc_info.split('_')[::2][::-1]]
            image = np.asarray(wsi.read_region((l, u), level, (size, size)))[:, :, :3]
            Image.fromarray(image).save(path)

        if self.preprocess_type == 'normal':
            image = image.transpose((2, 0, 1))
            image /= 255

        elif self.preprocess_type == 'vgg':
            image = image[:, :, ::-1].astype('float32')
            image -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
            image = image.transpose((2, 0, 1))
        else:
            raise ValueError('invalid preprocess type')

        return image
