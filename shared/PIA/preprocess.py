import random
import cv2
from skimage.color import rgb2hed
from PIL import Image, ImageMath, ImageEnhance, ImageFilter
import numpy as np


# import tensorflow as tf
# x = tf.placeholder(tf.uint8, shape=(256, 256, 3))
# img_ = tf.image.random_hue(x, 0.1)
# img_ = tf.image.random_brightness(img_, 48 / 255)
# img_ = tf.image.random_saturation(img_, 0.75, 1.25)
# img_ = tf.image.random_contrast(img_, 0.75, 1.25)


def color_aug(image):
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


def gaussian_blur(image):
    sigma = np.random.uniform(0.0, 3.0)
    img = image.filter(ImageFilter.GaussianBlur(sigma))
    return img


def preprocess(image, crop_size, color_augmentation=False, rotate=False, hed=False, nocrop=False):
    """
    :param image: PIL image
    :param image_size: size of the image
    :param crop_size: size of the processed image
    :param rotate: if True, rotate image as Data Augmentation
    :param color_augmentation: if True, perturb images with tensorflow
    :param hed: if True, transfer to hed space
    :param nocrop: if True, no cropping
    :param divide: it True, divide 255
    :return:
    """

    # color augmentation
    if color_augmentation:
        image = color_aug(image)

    # rotation (degree, not radian)
    if rotate:
        r = np.random.choice([0, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        image = image.transpose(r) if r > 0 else image

    image = np.asarray(image)

    # rgb2hed
    if hed:
        image = rgb2hed(image)

    # transpose
    image = image[:, :, ::-1].astype('float32')
    image -= np.array([104.0, 117.0, 123.0], dtype=np.float32)  # BGR
    image = image.transpose((2, 0, 1))

    # crop
    if not nocrop:
        crop_size = crop_size
        _, h, w = image.shape
        if random:
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

    return image
