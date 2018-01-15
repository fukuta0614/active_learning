from PIL import Image
import numpy as np


def preprocess(image: Image, type='googlenet'):
    image = np.asarray(image, dtype=np.float32)

    if type == 'fukuta':
        image /= 255.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (image - mean) / std
    else:
        image = image[:, :, ::-1]
        if 'resnet' in type:
            image -= np.array([103.063, 115.903, 123.152], dtype=np.float32)
        elif 'vgg' in type:
            image -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
        elif 'googlenet' in type:
            image -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
        else:
            raise ValueError('invalid type')

    image = image.transpose((2, 0, 1))
    return image
