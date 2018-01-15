import chainer
from PIL import Image
import numpy as np
import os
import random
from .base import tif_load
from .base import extracted_16_dir, extracted_17_dir, extracted_16_512_dir, extracted_17_512_dir
from skimage.color import rgb2hed
from PIL import Image, ImageMath, ImageEnhance, ImageFilter


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, crop_size, aug, color_aug, preprocess_type, texture):
        self.crop_size = crop_size
        self.aug = aug
        self.color_aug = color_aug
        self.preprocess_type = preprocess_type
        self.texture = texture

    def random_crop(self, image: Image):
        org_w, org_h = image.size
        # aspect = random.uniform(4 / 5., 5 / 4.)
        #
        # if aspect < 1.0:
        #     w = int(np.random.choice([0.8, 0.9, 1.0]) * org_w)
        #     h = int(w * aspect)
        # else:
        #     h = int(np.random.choice([0.8, 0.9, 1.0]) * org_h)
        #     w = int(h / aspect)
        h = self.crop_size
        w = self.crop_size
        top = random.randint(0, org_h - h)
        left = random.randint(0, org_w - w)

        image = image.crop((left, top, left + w, top + h))
        # image = image.resize((self.crop_size, self.crop_size), resample=Image.BICUBIC)

        return image

    def center_crop(self, image: Image):
        if self.texture:
            return image
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
        sigma = np.random.uniform(0.0, 2.0)  # 3.0
        img = image.filter(ImageFilter.GaussianBlur(sigma))
        return img

    def color_augment(self, image):
        # 色相いじる
        h, s, v = image.convert("HSV").split()
        _h = ImageMath.eval("(h + {}) % 255".format(np.random.randint(-25, 25)), h=h).convert("L")
        img = Image.merge("HSV", (_h, s, v)).convert("RGB")

        min_, max_ = 0.75, 1.25
        # 彩度を変える
        saturation_converter = ImageEnhance.Color(img)
        img = saturation_converter.enhance(np.random.uniform(min_, max_))

        # コントラストを変える
        contrast_converter = ImageEnhance.Contrast(img)
        img = contrast_converter.enhance(np.random.uniform(min_, max_))

        # 明度を変える
        brightness_converter = ImageEnhance.Brightness(img)
        img = brightness_converter.enhance(np.random.uniform(min_, max_))

        # シャープネスを変える
        sharpness_converter = ImageEnhance.Sharpness(img)
        img = sharpness_converter.enhance(np.random.uniform(min_, max_))

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
            # image = self.gaussian_blur(image)

        # 4. preprocess PIL -> numpy array, subtract or divide
        image = self.preprocess(image)

        # 5. Holizontal flip
        if random.randint(0, 1) and self.aug:
            image = image[:, :, ::-1]

        return image


class CamelyonDataset(PreprocessedDataset):
    def __init__(self, base, original_size=256, crop_size=224, aug=True, color_aug=False,
                 num_class=2, from_tif=False, preprocess_type='googlenet', texture=False):

        super(CamelyonDataset, self).__init__(crop_size, aug, color_aug, preprocess_type, texture)
        self.base = base
        self.image_size = original_size
        self.num_class = num_class
        self.from_tif = from_tif

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

        # load data
        x = self.base[i]

        # label
        if self.num_class == 2:
            label = np.int32(1 if x.split('_')[-1] == 'tumor' else 0)
        else:
            if x.split('_')[-1] == 'normal':
                label = np.int32(0)
            elif x.split('_')[-1] == 'tumor':
                label = np.int32(1)
            elif x.split('_')[-1] == 'middle':
                label = np.int32(2)
            else:
                raise ValueError('invalid label')

        # image
        if 'patient' in x:
            slide_name, (l, u) = '_'.join(x.split('_')[:4]), [int(_) for _ in x.split('_')[8:10]]
            if self.image_size == 256:
                path = os.path.join(extracted_17_dir, slide_name, x + '.png')
            else:
                path = os.path.join(extracted_17_512_dir, slide_name, x + '.png')
        else:
            slide_name, (l, u) = '_'.join(x.split('_')[:2]), [int(_) for _ in x.split('_')[6:8]]
            if self.image_size == 256:
                path = os.path.join(extracted_16_dir, slide_name, x.replace('middle', 'normal') + '.png')
            else:
                path = os.path.join(extracted_16_512_dir, slide_name, x.replace('middle', 'normal') + '.png')

        if not self.from_tif and os.path.exists(path):
            try:
                image = Image.open(path)
            except OSError as e:
                os.remove(path)
                print(path, e)
                return np.zeros((3, self.crop_size, self.crop_size)).astype('float32'), np.int32(-1)
        else:
            print('read', path)
            tif = tif_load(slide_name)
            image = tif.read_region((l, u), 0, (self.image_size, self.image_size))
            image = Image.fromarray(np.asarray(image)[:, :, :3])
            if not self.from_tif:
                image.save(path)
        try:
            image = self.process(image)
            return image, label

        except (OSError, ValueError) as e:
            os.remove(path)
            print(path, e)
            return np.zeros((3, self.crop_size, self.crop_size)).astype('float32'), np.int32(-1)


class CamelyonDatasetFromTif(PreprocessedDataset):
    def __init__(self, datasets, original_size=256, crop_size=224, aug=True, color_aug=False,
                 preprocess_type='googlenet', texture=False):

        super(CamelyonDatasetFromTif, self).__init__(crop_size, aug, color_aug, preprocess_type, texture)
        self.datasets = datasets
        self.image_size = original_size

    def __len__(self):
        # (6078267 + 272392 = 6350659)
        return 6350659

    def get_example(self, i):

        # load data
        label = random.choice(list(self.datasets))
        slide_name = random.choice(list(self.datasets[label]))
        _, _, _, _, l, u = random.choice(self.datasets[label][slide_name])

        try:
            tif = tif_load(slide_name)
            image = tif.read_region((l, u), 0, (self.image_size, self.image_size))

            image = self.process(image)
            label = np.int32(label == 'tumor')
            return image, label
        except (OSError, ValueError) as e:
            print(label, slide_name, (l, u))
            print(e)
            return np.zeros((3, self.crop_size, self.crop_size)).astype('float32'), np.int32(-1)


class CamelyonDatasetEx(PreprocessedDataset):
    """
    今まで各xが文字列で毎回parseしてたけどtupleにした
    """

    def __init__(self, base, original_size=256, crop_size=224, aug=True, color_aug=False,
                 preprocess_type='googlenet', texture=False):

        super(CamelyonDatasetEx, self).__init__(crop_size, aug, color_aug, preprocess_type, texture)
        self.base = base
        self.image_size = original_size

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

        # load data
        label, slide_name, patch_info = self.base[i]
        _, _, _, _, l, u = patch_info
        file_name = '{}_{}_{}.png'.format(slide_name, '_'.join([str(_) for _ in patch_info]), label)
        path = os.path.join(extracted_16_dir, slide_name, file_name)

        if os.path.exists(path):
            try:
                # print('exist')
                image = Image.open(path)
            except OSError as e:
                os.remove(path)
                print(path, e)
                return np.zeros((3, self.crop_size, self.crop_size)).astype('float32'), np.int32(0)
        else:
            try:
                # print('read')
                tif = tif_load(slide_name)
                image = tif.read_region((l, u), 0, (self.image_size, self.image_size))
                image = image.convert("RGB")
                image.save(path)
            except Exception as e:
                print(slide_name, e)
                return np.zeros((3, self.crop_size, self.crop_size)).astype('float32'), np.int32(0)
        try:
            image = self.process(image)
            label = np.int32(label == 'tumor')
            return image, label

        except (OSError, ValueError) as e:
            os.remove(path)
            print(path, e)
            return np.zeros((3, self.crop_size, self.crop_size)).astype('float32'), np.int32(0)


class CamelyonDatasetSaver(chainer.dataset.DatasetMixin):
    def __init__(self, base, original_size=256):
        self.base = base
        self.image_size = original_size

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

        # load data
        x = self.base[i]

        if 'patient' in x:
            slide_name, (l, u) = '_'.join(x.split('_')[:4]), [int(_) for _ in x.split('_')[8:10]]
        else:
            slide_name, (l, u) = '_'.join(x.split('_')[:2]), [int(_) for _ in x.split('_')[6:8]]

        tif = tif_load(slide_name)
        image = tif.read_region((l, u), 0, (self.image_size, self.image_size))
        image = np.asarray(image)[:, :, :3]

        if 'patient' in slide_name:
            path = os.path.join(extracted_17_dir, x + '.png')
        else:
            path = os.path.join(extracted_16_dir, x + '.png')
        Image.fromarray(image).save(path)

        return np.int32(1), np.int32(1)
