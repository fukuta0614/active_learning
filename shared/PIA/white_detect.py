from skimage import measure
import numpy as np


def is_white_old(im, local_cond=0.8):
    ave = (im > 200).mean()
    if ave > 0.7:
        return True

    elif ave > 0.5:
        image_size = im.shape[0]

        for i in range(3):
            for j in range(3):
                l, u = (image_size // 4) * i, (image_size // 4) * j
                if (im[l:l + image_size // 2, u:u + image_size // 2] > 200).mean() > local_cond:
                    return True
        else:
            return False
    else:
        return False


def is_white(im, global_cond=0.75, local_cond=0.25):
    ave = (im > 200).mean()
    if ave > global_cond:
        return True
    else:
        return calc_white_area(im) > local_cond


def calc_white_area(im, threshold=0.05):
    mask = measure.label(im > 200)
    props = measure.regionprops(mask)
    white = 0
    for i in range(np.amax(mask)):
        a = props[i].area / mask.shape[0] / mask.shape[1]
        if a > threshold:
            white += a

    return white
