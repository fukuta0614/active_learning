from itertools import combinations
import os
import cv2
import numpy as np
import openslide
from PIL import Image as PIL
from PIL import ImageDraw
from scipy import ndimage as nd
from skimage.filters import gaussian
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage import measure

from .base import tif_load, mask_tif_load, show_images_from_tif, get_bounding_box


def show_mistake(slide, wrong_imgs, target='17', image_size=256):
    print(show_images_from_tif)
    normal = [x for x in wrong_imgs if x.split('_')[-1] == 'normal']
    tumor = [x for x in wrong_imgs if x.split('_')[-1] == 'tumor']

    if len(normal) == 0:
        print('no mistake (normal)')
    else:
        print('normal image (predict tumor)')
        show_images_from_tif(slide, normal, target, image_size)

    if len(tumor) == 0:
        print('no mistake or no tumor\n')
    else:
        print('tumor image (predict normal)')
        show_images_from_tif(slide, tumor, target, image_size)


def show_patches_from_slides():
    base_dir = '/data/unagi0/fukuta/CAMELYON17/'
    heatmap_dir = os.path.join(base_dir, 'heatmap_512_texture')
    heatmap = {x: sorted(os.listdir(os.path.join(heatmap_dir, x))) for x in os.listdir(heatmap_dir) if
               len(os.listdir(os.path.join(heatmap_dir, x))) > 0}

    for slide in sorted(heatmap):
        patient = '_'.join(slide.split('_')[:2])

        print(slide)
        valid_npy = os.path.join(base_dir, 'Valid_Patch_Final_512', patient, slide + '.npy')
        patches = np.load(valid_npy)
        images = np.random.choice(patches, size=16)

        show_images_from_tif(slide, images, 'camelyon17')


def get_wrong_image(slide, heatmap_dir, valid_dir, ans_dict, base_dir='/data/unagi0/fukuta'):
    patient = '_'.join(slide.split('_')[:2])

    pred, tumor = os.listdir(os.path.join(base_dir, 'CAMELYON17', heatmap_dir, slide))
    predict_map = np.load(os.path.join(base_dir, 'CAMELYON17', heatmap_dir, slide, pred))
    valid_npy = os.path.join(base_dir, valid_dir, patient, slide + '.npy')

    patches = np.load(valid_npy)

    wrong_img = []
    if ans_dict[slide + '.tif'] == 'negative':
        for path in patches:
            i, j = [int(x) for x in path.split('.')[0].split('_')[:3:2]]
            if predict_map[i][j] > 0.75:
                wrong_img.append(slide + '_' + path + '_normal')

    elif isinstance(patches[0].split('_')[-1], str):
        for path in patches:
            i, j = [int(x) for x in path.split('.')[0].split('_')[:3:2]]
            if path.split('_')[-1] == 'normal' and predict_map[i][j] > 0.75:
                wrong_img.append(slide + '_' + path)
            elif path.split('_')[-1] == 'tumor' and predict_map[i][j] < 0.9:
                wrong_img.append(slide + '_' + path)
            else:
                pass
    else:
        pass

    return wrong_img


def make_evaluation_mask(slide_name, visualize=True, ans_dict=None, resolution=0.25, level=5):
    import matplotlib.pyplot as plt

    tif = mask_tif_load(slide_name)
    img = tif.read_region((0, 0), 5, tif.level_dimensions[5])

    pixelarray = np.array(img)
    distance = nd.distance_transform_edt(255 - pixelarray[:, :, 0])
    Threshold = 75 / (resolution * pow(2, level) * 2)  # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity=2)

    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold1 = 275 / (resolution * pow(2, level))
    threshold2 = 2075 / (resolution * pow(2, level))
    threshold3 = 3075 / (resolution * pow(2, level))

    ans = 'itc'
    max_length = 0
    for i in range(0, max_label):
        major_axis_length = properties[i].major_axis_length
        max_length = max(major_axis_length, max_length)
        if major_axis_length < threshold1:
            pass
        elif major_axis_length < threshold2:
            ans = 'Micro' if ans == 'itc' else ans
        # evaluation_mask[evaluation_mask == (i+1)] =
        elif major_axis_length < threshold3:
            ans = 'Macro'
        else:
            ans = 'Deka Macro'
            break

            #     if ans != ans_dict[slide_name]:
            #         for i in range(0, max_label):
            #             print(properties[i].major_axis_length * (resolution * pow(2, level)))
    print(ans, '-', max_length * (resolution * pow(2, level)))
    if visualize:
        plt.figure(figsize=(12, 12))
        plt.imshow(evaluation_mask)
        plt.show()
    return slide_name, ans, max_length * (resolution * pow(2, level))


def detect_tissue_region(path):
    """
    WSI元画像, detected region, 輪郭囲う
    """
    import matplotlib.pyplot as plt

    slide_name = path.split('/')[-1].split('.')[0]
    tif = openslide.OpenSlide(path)

    region = tif.read_region((0, 0), 5, tif.level_dimensions[5])

    # hsv空間で大津の二値化
    region = np.array(region)[:, :, :3]
    black = np.stack([region.mean(axis=2) < 10] * 3).transpose(1, 2, 0)
    region = black * np.ones(region.shape).astype('uint8') * 255 + ~black * region
    mean_cond = region.mean(axis=2) > 50
    hsv = cv2.cvtColor(np.array(region), cv2.COLOR_BGR2HSV)
    ret, th1 = cv2.threshold(hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, th2 = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_ = (th1 > 0) * (th2 > 0) * mean_cond
    new_mask = gaussian(mask_, sigma=0.3) > 0.01
    new_mask = median(new_mask, disk(12)) == 255

    fig, axes = plt.subplots(1, 3, figsize=(16, 16))

    # 元画像
    region = tif.read_region((0, 0), 5, tif.level_dimensions[5])
    region = np.array(region)[:, :, :3]
    axes[0].imshow(region)
    axes[0].axis('off')

    # 黒白 除く
    axes[1].imshow(np.stack([new_mask] * 3).transpose(1, 2, 0) * region
                   + (np.stack([~new_mask] * 3).transpose(1, 2, 0) * np.ones(region.shape) * 128).astype('uint8'))
    axes[1].axis('off')

    # 輪郭抽出
    contours = measure.find_contours(new_mask, 0.8)
    axes[2].axis('off')
    axes[2].imshow(region)
    for n, contour in enumerate(contours):
        axes[2].plot(contour[:, 1], contour[:, 0], '-b', linewidth=1)
    axes[2].set_xlim(0, region.shape[1])
    axes[2].set_ylim(region.shape[0], 0)
    plt.show()


def visualize_annotation(slide, anno_slide, black_remove=True, overall=True, whole_show=False, box_show=True,
                         patch_show=True,
                         image_size=256, patch_level=1):
    import matplotlib.pyplot as plt

    slide_name = slide.split('/')[-1].split('.')[0]
    tif = openslide.OpenSlide(slide)
    cancer_tif = openslide.OpenSlide(anno_slide)

    if whole_show:
        plt.figure(figsize=(16, 16))
        plt.imshow(tif.read_region((0, 0), 5, tif.level_dimensions[5]))
        plt.show()

    # hsv空間で大津の二値化
    max_level = len(cancer_tif.level_dimensions) - 1
    region = tif.read_region((0, 0), max_level, tif.level_dimensions[max_level])
    if black_remove:
        black = np.stack([region.mean(axis=2) < 10] * 3).transpose(1, 2, 0)
        region = black * np.ones(region.shape).astype('uint8') * 255 + ~black * region
    mean_cond = np.array(region)[:, :, :3].mean(axis=2) > 50
    hsv = cv2.cvtColor(np.array(region), cv2.COLOR_BGR2HSV)
    ret, th1 = cv2.threshold(hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, th2 = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_ = (th1 > 0) * (th2 > 0) * mean_cond
    new_mask = gaussian(mask_, sigma=0.3) > 0.01

    # level 全体
    region = tif.read_region((0, 0), max_level, tif.level_dimensions[max_level])

    region_mask = cancer_tif.read_region((0, 0), max_level, cancer_tif.level_dimensions[max_level])
    region_mask_ = np.array(region_mask)[:, :, :3].mean(axis=2) > 0
    cancer_boxes = get_bounding_box(region_mask_)

    # levelにmaskを合わせる
    if overall:
        fig, axes = plt.subplots(1, 3, figsize=(16, 16))
        region = np.array(region)[:, :, :3]
        axes[1].imshow(np.stack([new_mask] * 3).transpose(1, 2, 0) * region
                       + (np.stack([~new_mask] * 3).transpose(1, 2, 0) * np.ones(region.shape) * 128).astype('uint8'))
        axes[2].imshow(np.array(region) * (np.array(region_mask)[:, :, :3] > 0))

        region = tif.read_region((0, 0), len(cancer_tif.level_dimensions) - 1, cancer_tif.level_dimensions[-1])
        for cancer_box in cancer_boxes:
            dr = ImageDraw.Draw(region)
            dr.rectangle(cancer_box, outline=(0, 0, 128))
        axes[0].imshow(region)

        plt.show()

    # cancer_box
    if box_show:
        for cancer_box in cancer_boxes:
            print(cancer_box, end=' - ')

            fig, axes = plt.subplots(1, 3, figsize=(16, 16))
            left, up, right, down = cancer_box[:4]

            max_level = 5

            cancer_zero_ratio = np.array(cancer_tif.level_dimensions[0]) // np.array(cancer_tif.level_dimensions[-1])
            cancer_level_ratio = np.array(cancer_tif.level_dimensions[max_level]) // np.array(
                cancer_tif.level_dimensions[-1])

            w, h = (right - left, down - up) * cancer_level_ratio
            print((right - left, down - up) * cancer_zero_ratio)

            region = np.array(tif.read_region((left, up) * cancer_zero_ratio, max_level, (w, h)))
            region_mask = np.array(cancer_tif.read_region((left, up) * cancer_zero_ratio, max_level, (w, h)))

            axes[0].imshow(region)
            axes[1].imshow(region[:, :, :3] * (region_mask[:, :, :3] == 0))
            contours = measure.find_contours(region_mask[:, :, 0], 0.8)
            axes[2].axis('off')
            axes[2].imshow(region)
            for n, contour in enumerate(contours):
                axes[2].plot(contour[:, 1], contour[:, 0], '-b', linewidth=2)
            axes[2].set_xlim(0, region.shape[1])
            axes[2].set_ylim(region.shape[0], 0)
            plt.show()
            # axes[2].imshow(region * (region_mask > 0))
            # plt.show()

            if patch_show:
                max_level = patch_level
                cancer_zero_ratio = np.array(cancer_tif.level_dimensions[0]) // np.array(
                    cancer_tif.level_dimensions[-1])
                cancer_level_ratio = np.array(cancer_tif.level_dimensions[max_level]) // np.array(
                    cancer_tif.level_dimensions[-1])
                level_zero_ratio = np.array(tif.level_dimensions[0]) // np.array(tif.level_dimensions[max_level])
                w, h = (right - left, down - up) * cancer_level_ratio

                w_num, h_num, w_offset, h_offset = w // image_size, h // image_size, w % image_size // 2, h % image_size // 2
                print(w_num, h_num)
                fig = plt.figure(figsize=(6, 6 * h // w))
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

                overlap = 0

                for i in range(h_num):
                    for j in range(w_num):
                        l = w_offset + j * image_size
                        u = h_offset + i * image_size
                        start = (left, up) * cancer_zero_ratio + (l, u) * level_zero_ratio

                        region = tif.read_region(start, max_level, (image_size, image_size))
                        cancer_region = cancer_tif.read_region(start, max_level, (image_size, image_size))

                        ax = fig.add_subplot(h_num, w_num, w_num * i + j + 1, xticks=[], yticks=[])
                        if (np.asarray(cancer_region)[:, :, :3].mean(axis=2) == 0).mean() > 0.6:
                            ax.imshow(128 * np.ones((image_size, image_size, 3)))
                        else:
                            ax.imshow(region)
                plt.show()

                fig = plt.figure(figsize=(6, 6 * h // w))
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

                for i in range(h_num):
                    for j in range(w_num):
                        l = w_offset + j * (image_size - overlap)
                        u = h_offset + i * (image_size - overlap)
                        start = (left, up) * cancer_zero_ratio + (l, u) * level_zero_ratio

                        region = tif.read_region(start, max_level, (image_size, image_size))
                        cancer_region = cancer_tif.read_region(start, max_level, (image_size, image_size))

                        img_normal = PIL.fromarray(
                            np.array(region)[:, :, :3] * (np.array(cancer_region)[:, :, :3] == 0))

                        ax = fig.add_subplot(h_num, w_num, w_num * i + j + 1, xticks=[], yticks=[])
                        if (np.asarray(img_normal)[:, :, :3].mean(axis=2) < 10).mean() > 0.2 or (
                                    np.asarray(img_normal).mean(axis=2) > 220).mean() > 0.4:
                            ax.imshow(128 * np.ones((image_size, image_size, 3)))
                        else:
                            ax.imshow(img_normal)
                plt.show()


def visualize_vaild(valid_npy, base_dir='/data/unagi0/fukuta/'):
    """
    valid patch 抽出結果
    """
    import matplotlib.pyplot as plt
    # 閾値処理後
    slide = valid_npy.split('/')[-1].split('.')[0]
    patches = np.load(valid_npy)
    tif = tif_load(slide, base_dir)

    height, width = [int(x) for x in patches[0].split('_')[1:4:2]]
    img = np.zeros((height, width)) == 1
    for path in patches:
        i, j = [int(x) for x in path.split('.')[0].split('_')[:3:2]]
        img[i][j] = True

    fig, axes = plt.subplots(1, 3, figsize=(16, 16))
    thumb = tif.read_region((0, 0), 8, tif.level_dimensions[8])
    axes[0].imshow(thumb)
    axes[1].imshow(img, cmap=plt.cm.gray)
    axes[2].imshow(np.stack([img] * 3).transpose(1, 2, 0) * np.array(thumb)[:, :, :3])
    plt.show()


def visualize_vaild_with_annotation(valid_npy, slide_ans=None, base_dir='/data/unagi0/fukuta/'):
    import matplotlib.pyplot as plt

    patches = np.load(valid_npy)
    slide = valid_npy.split('/')[-1].split('.')[0]

    cancer_tif = mask_tif_load(slide, base_dir)
    if cancer_tif is None:
        return

    print(slide, end=' - ')
    if slide_ans is not None:
        print(slide_ans[slide.split('.')[0] + '.tif'], end=' - ')
    print(len([x for x in patches if 'tumor' in x]), 'patches')

    tif = tif_load(slide, base_dir)

    height, width = [int(x) for x in patches[0].split('_')[1:4:2]]
    img = np.zeros((height, width)) == 1
    img_tumor = np.zeros((height, width)) == 1
    for path in patches:
        i, j = [int(x) for x in path.split('.')[0].split('_')[:3:2]]
        img[i][j] = True
        if path.split('_')[-1] == 'tumor':
            img_tumor[i][j] = True

    fig, axes = plt.subplots(1, 4, figsize=(16, 16))
    thumb = tif.read_region((0, 0), 8, tif.level_dimensions[8])
    try:
        cancer_thumb = cancer_tif.read_region((0, 0), 8, cancer_tif.level_dimensions[8])
    except IndexError:
        cancer_thumb = cancer_tif.read_region((0, 0), 6, cancer_tif.level_dimensions[6])
        cancer_thumb = np.array(cancer_thumb.resize(img_tumor.shape[::-1]))[:, :, :3]

    axes[0].imshow(thumb)
    axes[1].imshow(np.stack([img] * 3).transpose(1, 2, 0) * np.array(thumb)[:, :, :3])
    axes[2].imshow(cancer_thumb)
    axes[3].imshow(img_tumor, cmap=plt.cm.gray)
    plt.show()


def overlay_heatmap(image, heatmap):
    import matplotlib.pyplot as plt
    import mpl_toolkits.axes_grid1
    fig, ax = plt.subplots()
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(heatmap)
    ax.imshow(image, alpha=0.5)
    plt.colorbar(im, cax)
    plt.show()
