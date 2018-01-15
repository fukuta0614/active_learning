import math
import os
import pickle
from collections import Counter
from itertools import combinations

import numpy as np
import openslide
import pandas as pd

from sklearn.metrics import cohen_kappa_score
from skimage import measure

server = os.uname()[1]
if 'usropsai05' in server:  ## for amed
    camelyon17 = '/home/users/usropsai05/Camelyon/CAMELYON17'
    camelyon16 = '/home/users/usropsai05/Camelyon/CAMELYON16'

    base_dir_16 = '/home/users/usropsai05/Camelyon'
    base_dir_17 = '/home/users/usropsai05/Camelyon'

    home_dir = '/home/users/usropsai05/work_space/Camelyon/'

elif server == 'dl-box-docker':
    camelyon17 = '/data/8T-HDD-2/CAMELYON17'
    camelyon16 = '/data/8T-HDD-2/CAMELYON16'

    base_dir_16 = '/data/Camelyon16'
    base_dir_17 = '/data/Camelyon17'

    home_dir = '/home/fukuta/work_space/Camelyon/'
elif server == 'kali-docker':
    camelyon17 = '/mnt/8T-HDD-2/CAMELYON17'
    camelyon16 = '/mnt/8T-HDD-2/CAMELYON16'

    base_dir_16 = '/data1/Camelyon16'
    base_dir_17 = '/data1/Camelyon17'

    home_dir = '/home/fukuta/work_space/Camelyon/'
else:
    ugui = '/data/ugui0/fukuta/'
    unagi = '/data/unagi0/fukuta'
    camelyon17 = os.path.join(unagi, 'CAMELYON17')
    camelyon16 = os.path.join(unagi, 'CAMELYON16')

    base_dir_16 = os.path.join(ugui, 'Camelyon16')
    base_dir_17 = os.path.join(ugui, 'Camelyon17')

    home_dir = '/home/mil/fukuta/work_space/pathology/Camelyon/'
    slide_diag_ans_path = os.path.join(home_dir, 'data/Camelyon/slide_diag_ans.pkl')

tif_16_train_normal = os.path.join(camelyon16, 'TrainingData/Train_Normal/')
tif_16_train_tumor = os.path.join(camelyon16, 'TrainingData/Train_Tumor/')
tif_16_test = os.path.join(camelyon16, 'Testset/Images/')
tif_17_train = os.path.join(camelyon17, 'training/images/')
tif_17_test = os.path.join(camelyon17, 'testing/images/')

extracted_16_dir = os.path.join(base_dir_16, 'Dataset16')
extracted_17_dir = os.path.join(base_dir_17, 'Dataset17')
extracted_16_512_dir = os.path.join(base_dir_16, 'Dataset16_512')
extracted_17_512_dir = os.path.join(base_dir_17, 'Dataset17_512')

if 'usropsai05' in server:  # TODO(今だけ)
    extracted_16_dir = '/home/users/usropsai05/Camelyon/Dataset16/'
    camelyon16 = '/home/users/usropsai05/Camelyon'
    tif_16_train_normal = os.path.join(camelyon16, 'Train_Normal/')
    tif_16_train_tumor = os.path.join(camelyon16, 'Train_Tumor/')
    tif_16_test = os.path.join(camelyon16, 'Images/')

dataset_path = os.path.join(home_dir, 'dataset')

have_annotation = []
if os.path.exists(os.path.join(camelyon17)) and os.path.exists(os.path.join(camelyon16)):
    for x in os.listdir(os.path.join(camelyon17, 'training/lesion_annotations/')):
        have_annotation.append(x.split('.xml')[0] + '.tif')

    csv = pd.read_csv(os.path.join(camelyon17, 'training/stage_labels.csv'))
    patient_ans = {a.split('.')[0]: b for a, b in csv.ix[np.arange(100) * 6].values}

    slide_ans_17 = {}
    for a, b in csv.values:
        if a.endswith('.tif'):
            slide_ans_17[a] = b

    with open(os.path.join(camelyon16, 'slide_diag_ans.pkl'), 'rb') as f:
        slide_ans_16 = pickle.load(f)


def mask_tif_load(slide_name):
    path = os.path.join(camelyon16, 'TrainingData/Ground_Truth/Mask/{}_Mask.tif'.format(slide_name))
    if os.path.exists(path):
        return openslide.OpenSlide(path)

    path = os.path.join(camelyon16, 'Testset/Ground_Truth/Masks/{}_Mask.tif'.format(slide_name))
    if os.path.exists(path):
        return openslide.OpenSlide(path)

    path = os.path.join(camelyon17, 'training/lesion_annotations', '{}_mask.tif'.format(slide_name))
    if os.path.exists(path):
        return openslide.OpenSlide(path)
    return None


def tif_load(slide_name):
    if 'Tumor' in slide_name:
        tif_path = os.path.join(tif_16_train_tumor, slide_name + '.tif')
    elif 'Normal' in slide_name:
        tif_path = os.path.join(tif_16_train_normal, slide_name + '.tif')
    elif 'Test' in slide_name:
        tif_path = os.path.join(tif_16_test, slide_name + '.tif')
    elif 'patient' in slide_name:
        patient = '_'.join(slide_name.split('_')[:2])
        if int(patient.split('_')[1]) < 100:
            tif_path = os.path.join(tif_17_train, patient, slide_name + '.tif')
        else:
            tif_path = os.path.join(tif_17_test, patient, slide_name + '.tif')
    else:
        raise ValueError('invalid slidename')

    tif = openslide.OpenSlide(tif_path)
    return tif


def diagnose_metastasis(major_axis_length, level=8, resolution=0.25):
    threshold0 = 100 / (resolution * pow(2, level))
    threshold1 = 200 / (resolution * pow(2, level))
    threshold2 = 2000 / (resolution * pow(2, level))

    if major_axis_length < threshold0:
        return 'negative'
    elif major_axis_length < threshold1:
        return 'itc'
    if major_axis_length < threshold2:
        return 'micro'
    else:
        return 'macro'


def diagnose_patient(slide_ans):
    c = Counter(slide_ans)
    if c['negative'] == 5:
        return 'pN0'
    elif not 'micro' in c and not 'macro' in c:
        return 'pN0(i+)'
    elif not 'macro' in c:
        return 'pN1mi'
    elif c['micro'] + c['macro'] < 4:
        return 'pN1'
    elif c['micro'] + c['macro'] >= 4:
        return 'pN2'
    else:
        raise ValueError()


def calc_kappa_score(submission):
    if isinstance(submission, str):
        submission = pd.read_csv('submission_train.csv')
    else:
        submission = pd.DataFrame(submission)
    ground_truth = pd.read_csv('/mnt/8T-HDD-2/CAMELYON17/training/stage_labels.csv')

    stage_list = ['pN0', 'pN0(i+)', 'pN1mi', 'pN1', 'pN2']
    ground_truth_map = {df_row[0]: df_row[1] for _, df_row in ground_truth.iterrows() if
                        str(df_row[0]).lower().endswith('.zip')}
    submission_map = {df_row[0]: df_row[1] for _, df_row in submission.iterrows() if
                      str(df_row[0]).lower().endswith('.zip')}

    ground_truth_stage_list = []
    submission_stage_list = []
    for patient_id, ground_truth_stage in ground_truth_map.items():
        # Check consistency: all stages must be from the official stage list and there must be a submission for each patient in the ground truth.

        if ground_truth_stage not in stage_list:
            raise ValueError('Unknown stage in ground truth: {stage}'.format(stage=ground_truth_stage))
        if patient_id not in submission_map:
            raise ValueError('Patient missing from submission: {patient}'.format(patient=patient_id))
        if submission_map[patient_id] not in stage_list:
            raise ValueError('Unknown stage in submission: {stage}'.format(stage=submission_map[patient_id]))

        # Add the pair to the lists.
        #
        ground_truth_stage_list.append(ground_truth_stage)
        submission_stage_list.append(submission_map[patient_id])
    return cohen_kappa_score(y1=ground_truth_stage_list, y2=submission_stage_list, labels=stage_list,
                             weights='quadratic')


def show_images_from_tif(slide, images, target='camelyon17', image_size=256, labels=None):
    import matplotlib.pyplot as plt

    num = len(images)
    n = math.sqrt(num)
    assert n == int(n)
    tif = tif_load(slide)
    if labels:
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)
    else:
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    if len(images) > num:
        images = np.random.choice(images, size=num, replace=False)

    count = 0
    for image in images:
        count += 1
        if target == 'camelyon16':
            l, u = [int(x) for x in image.split('_')[6:8]]
        elif target == 'camelyon17':
            l, u = [int(x) for x in image.split('_')[8:10]]
        else:
            raise ValueError()
        region = tif.read_region((l, u), 0, (image_size, image_size))
        ax = fig.add_subplot(n, n, count, xticks=[], yticks=[])
        ax.imshow(region)
    plt.show()


def get_bounding_box(image, merge=True, margin=5):
    """
    :param image:
    :param merge:
    :param margin:
    :return:
    """
    # segmentation
    height, width = image.shape
    seg = np.zeros(image.shape).astype('int32')
    same = {0: 0}
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if image[i][j]:
                surround = set([seg[x][y] for (x, y) in (
                    (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j),
                    (i + 1, j + 1)) if seg[x][y]])
                if len(surround) == 0:
                    seg[i][j] = seg.max() + 1
                    same[seg[i][j]] = seg[i][j]
                elif len(surround) == 1:
                    seg[i][j] = min(surround)
                else:
                    mini = min(surround)
                    seg[i][j] = mini
                    for s in surround:
                        same[s] = mini

    # merge same segment
    for k in same:
        num = k
        while True:
            if same[num] == num:
                break
            else:
                num = same[num]
        same[k] = num

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            seg[i][j] = same[seg[i][j]]

    # calculate box and delete small box
    boxes = []
    for i in sorted(set(seg.ravel()))[1:]:
        seg_ = (seg == i)
        h = np.arange(height)[seg_.any(axis=1)]
        w = np.arange(width)[seg_.any(axis=0)]
        boxes.append((w.min(), h.min(), w.max(), h.max(), w.mean(), h.mean()))

    def center(box):
        return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2

    if merge:
        # decrease boxes to cell_num
        while True:
            for i, j in combinations(range(len(boxes)), 2):
                a, b = center(boxes[i])
                c, d = center(boxes[j])
                if np.sqrt((a - c) ** 2 + (d - b) ** 2) < 100:
                    a = boxes.pop(i)
                    b = boxes.pop(j - 1)
                    box = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), np.mean([a[4], b[4]]),
                           np.mean([a[5], b[5]]))
                    boxes.insert(i, box)
                    break
            else:
                break

    boxes = [(b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin) for b in boxes]

    return boxes


def extract_max_length(image, level=8, resolution=0.25):
    from scipy import ndimage as nd
    filled_image = nd.morphology.binary_fill_holes(image)
    evaluation_mask = measure.label(filled_image, connectivity=2)

    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)

    if max_label > 0:
        return (resolution * pow(2, level)) * max([properties[i].major_axis_length for i in range(0, max_label)])
    else:
        return 0
