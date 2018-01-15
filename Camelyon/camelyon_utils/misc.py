import matplotlib.pyplot as plt

from skimage import measure
import numpy as np
from scipy import ndimage as nd
from collections import Counter
import os
import openslide
import pickle
from xml.etree import ElementTree

from .base import camelyon16, camelyon17, tif_load, patient_ans, slide_ans_17


def use_xml():
    """
    annotation xml使ってみようとしたコード
    """
    tree = ElementTree.parse(os.path.join(camelyon16, 'TrainingData/Ground_Truth/XML/Tumor_{}.xml'.format(id)))
    tif = openslide.OpenSlide()
    level = 1
    coordinates = tree.getroot()[0][0][0]
    annotations_ = [[float(e.get('X')), float(e.get('Y'))] for e in coordinates]
    annotations = np.array(annotations_) * np.array(tif.level_dimensions[level]) / np.array(tif.level_dimensions[0])
    annotations = annotations.astype('int64')

    region = tif.read_region((0, 0), level, tif.level_dimensions[level])
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    axes[0].imshow(region)
    region = np.array(region)[:, :, :3]

    truth = np.zeros(region.shape)
    for c in annotations:
        truth[c[1], c[0], :] = True
    axes[1].imshow(truth)
    plt.show()


def copy_from_hdd_to_sdd():
    """
    HDDから内蔵HDDに移したコード (微妙 shutil使った方がよさそう）
    """
    with open(os.path.join(camelyon16, 'train_test.pkl'), 'rb') as f:
        train_data, test_data = pickle.load(f)

    for i, x in enumerate(train_data):
        if i % 10000 == 0:
            print(i)
        src_dir = '/media/8T-HDD/CAMELYON16/Extracted/level0-256/'
        dst_dir = '/home/fukuta/work_space/Camelyon/images/test/'
        src = os.path.join(src_dir, '_'.join(x.split('_')[:2]),
                           '_'.join(x.split('_')[2:6]) + '_' + x.split('_')[-1] + '.png')
        dst = os.path.join(dst_dir, x + '.png')
        if not os.path.exists(dst):
            os.system('cp {} {}'.format(src, dst))


def make_dataset_17():
    """
    17でannotationついてる50とnegative50をいい感じに選択
    """

    valid_patch_anno_dir = os.path.join(camelyon17, 'training/Valid_Patch_Annotation_256')
    valid_patch_dir = os.path.join(camelyon17, 'training/Valid_Patch_256')

    all_patients = set(list(patient_ans))
    pN0 = [a for a, b in patient_ans.items() if b == 'pN0']
    anno = os.listdir(valid_patch_anno_dir)
    anno_slide = [os.path.join(valid_patch_anno_dir, p, x) for p in os.listdir(valid_patch_anno_dir) for x in
                  os.listdir(os.path.join(valid_patch_anno_dir, p))]
    pN0_slide = [
        os.path.join(valid_patch_dir, p, np.random.choice([x for x in os.listdir(os.path.join(valid_patch_dir, p))]))
        for p in pN0
        ]
    res_negative_slide = []
    for p in list(set(all_patients) - set(pN0) - set(anno)):
        if len(res_negative_slide) < 26:
            negative = [x for x in os.listdir(os.path.join(valid_patch_dir, p)) if
                        slide_ans_17[x.replace('.npy', '.tif')] == 'negative']
            if len(negative) > 0:
                res_negative_slide.append(os.path.join(valid_patch_dir, p, np.random.choice(negative)))

    test_17 = anno_slide + pN0_slide + res_negative_slide

    np.save('test_17', np.array(test_17))


def show_result_of_white_thresholding():
    base_dir = '/data/unagi0/fukuta/CAMELYON17/'
    heatmap_dir = os.path.join(base_dir, 'heatmap_512_texture')
    heatmap = {x: sorted(os.listdir(os.path.join(heatmap_dir, x))) for x in os.listdir(heatmap_dir) if
               len(os.listdir(os.path.join(heatmap_dir, x))) > 0}

    for slide in sorted(heatmap):
        patient = '_'.join(slide.split('_')[:2])

        pred, tumor = heatmap[slide]
        print(slide)

        predict_map = np.load(os.path.join(heatmap_dir, slide, pred))
        predict_map2 = np.load(os.path.join(os.path.join(base_dir, 'heatmap_512_texture_backup'), slide, pred))
        valid_npy = os.path.join(base_dir, 'Valid_Patch_Final_512', patient, slide + '.npy')
        valid_npy2 = os.path.join(base_dir, 'valid_patch_512', patient, slide + '.npy')
        print((predict_map2 > 0).sum(), len(np.load(valid_npy2)), (predict_map > 0).sum(), len(np.load(valid_npy)))
        patches = np.load(valid_npy)
        patches2 = np.load(valid_npy2)
        ij2lu = {tuple(int(x) for x in path.split('.')[0].split('_')[:3:2]): tuple(
            int(x) for x in path.split('.')[0].split('_')[4:6])
                 for path in patches}
        ij2lu2 = {tuple(int(x) for x in path.split('.')[0].split('_')[:3:2]): tuple(
            int(x) for x in path.split('.')[0].split('_')[4:6])
                  for path in patches2}
        w, h = predict_map2.shape
        tif = tif_load(slide)
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        count = 0
        for i in range(w):
            for j in range(h):
                if predict_map2[i][j] > 0 and (i, j) not in ij2lu:
                    count += 1
                    region = tif.read_region(ij2lu2[(i, j)], 0, (512, 512))
                    ax = fig.add_subplot(8, 8, count, xticks=[], yticks=[])
                    ax.imshow(region)

                    if count == 64:
                        plt.show()
                        break
            else:
                continue
            break
        else:
            plt.show()
