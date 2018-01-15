import numpy as np
from scipy import ndimage as nd
from collections import Counter
import os
import openslide
from skimage import measure


def calc_map(w, h, image_size, overlap_rate=0.0):
    overlap = int(image_size * overlap_rate)
    w_num, h_num = (w - image_size) // (image_size - overlap) + 1, (h - image_size) // (image_size - overlap) + 1
    w_offset, h_offset = (w - image_size) % (image_size - overlap) // 2, (h - image_size) % (image_size - overlap) // 2
    return w_num, h_num, w_offset, h_offset


def add_heatmap(predict_map_512, predict_map_256):
    h, w = predict_map_512.shape
    predict_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if predict_map_256[i][j] > 0 and predict_map_512[i][j] > 0:
                predict_map[i][j] = (predict_map_256[i][j] + predict_map_512[i][j]) / 2
            elif predict_map_256[i][j] > 0:
                predict_map[i][j] = predict_map_256[i][j]
            elif predict_map_512[i][j] > 0:
                predict_map[i][j] = predict_map_512[i][j]
            else:
                predict_map[i][j] = 0

    return predict_map


def heatmap_hosei(predict_map_512, shape):
    h, w = shape
    h_512, w_512 = predict_map_512.shape
    assert (h, w) == (h_512 + 1, w_512 + 1)

    predict_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                predict_map[i][j] = predict_map_512[i][j]
            elif i == 0 and j == w - 1:
                predict_map[i][j] = predict_map_512[i][j - 1]
            elif i == h - 1 and j == 0:
                predict_map[i][j] = predict_map_512[i - 1][j]
            elif i == h - 1 and j == w - 1:
                predict_map[i][j] = predict_map_512[i - 1][j - 1]

            elif i == 0:
                predict_map[i][j] = (predict_map_512[i][j - 1] + predict_map_512[i][j]) / 2
            elif j == 0:
                predict_map[i][j] = (predict_map_512[i - 1][j] + predict_map_512[i][j]) / 2
            elif i == h - 1:
                predict_map[i][j] = (predict_map_512[i - 1][j - 1] + predict_map_512[i - 1][j]) / 2
            elif j == w - 1:
                predict_map[i][j] = (predict_map_512[i - 1][j - 1] + predict_map_512[i][j - 1]) / 2

            else:
                sum_ = [predict_map_512[x][y]
                        for x, y in [(i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)]
                        if predict_map_512[x][y] > 0]
                predict_map[i][j] = np.mean(sum_) if len(sum_) > 0 else 0
    return predict_map


def heatmap_hosei_zure(slide, base_dir='/data/unagi0/fukuta'):
    heatmap_dir_512 = os.path.join(base_dir, 'heatmap_512_texture_overlap')
    heatmap_dir_256 = os.path.join(base_dir, 'heatmap_256_resnet')
    heatmap = {x: sorted(os.listdir(os.path.join(heatmap_dir_256, x))) for x in os.listdir(heatmap_dir_256) if
               len(os.listdir(os.path.join(heatmap_dir_256, x))) > 0}
    pred, tumor = heatmap[slide]

    predict_map_256 = np.load(os.path.join(heatmap_dir_256, slide, pred))
    predict_map_512 = np.load(os.path.join(heatmap_dir_512, slide, pred))

    tif = tif_read(slide)
    w, h = tif.level_dimensions[0]
    w_num_256, h_num_256, w_offset_256, h_offset_256 = calc_map(w, h, 256)
    w_num_512, h_num_512, w_offset_512, h_offset_512 = calc_map(w, h, 512)

    predict_map = np.zeros((2 * h_num_256, 2 * w_num_256))
    for i in range(2 * h_num_256):
        u = i * 128 + h_offset_256
        i_256 = (u - h_offset_256) // 256
        i_512 = (u - h_offset_512) // 512
        for j in range(2 * w_num_256):
            l = j * 128 + w_offset_256
            j_256 = (l - w_offset_256) // 256
            j_512 = (l - w_offset_512) // 512

            pred_256 = predict_map_256[i_256][j_256]
            pred_512 = predict_map_512[i_512][j_512] if 0 <= i_512 < h_num_512 and 0 <= j_512 < w_num_512 else 0
            if pred_256 > 0 and pred_512 > 0:
                predict_map[i][j] = (pred_256 + pred_512) / 2
            elif pred_256 > 0:
                predict_map[i][j] = pred_256
            elif pred_512 > 0:
                predict_map[i][j] = pred_512
            else:
                pass
    return predict_map
