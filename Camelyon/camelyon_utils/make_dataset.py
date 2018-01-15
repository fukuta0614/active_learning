import os
import pickle
from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import tif_load


def extract_training_dataset_16(slide_diag_ans, base_dir='/data/unagi0/fukuta'):
    not_exhaustively_annotated = ['Tumor_015', 'Tumor_018', 'Tumor_020', 'Tumor_029', 'Tumor_033', 'Tumor_044',
                                  'Tumor_046', 'Tumor_051', 'Tumor_054', 'Tumor_055', 'Tumor_079', 'Tumor_092',
                                  'Tumor_095', 'Normal_086']
    not_all_tumor = ['Tumor_061', 'Tumor_082']
    not_valid = ['Test_049', 'Test_114', 'Normal_086']

    dataset = defaultdict(dict)

    valid_dir = os.path.join(base_dir, '/CAMELYON16/valid_patch_512_overlap/')

    exists = None
    # with open('/home/mil/fukuta/work_space/pathology/Camelyon/exists.pkl', 'rb') as f:
    #     exists = pickle.load(f)
    # for slide_name, images in exists.items():
    for x in sorted(os.listdir(valid_dir)):
        slide_name = x.split('.')[0]
        if slide_name in not_valid:
            continue

        diag = slide_diag_ans[slide_name]
        print(slide_name, diag, end=' - ')

        dataset[diag][slide_name] = {}
        if exists is None:
            images = np.load(os.path.join(valid_dir, x))
            images = [slide_name + '_' + image for image in images]
        else:
            images = [image.split('.')[0] for image in images]

        normal = [image for image in images if image.split('_')[-1] == 'normal']
        tumor = [image for image in images if image.split('_')[-1] == 'tumor']

        print('normal', len(normal), 'tumor', len(tumor))

        if diag == 'negative':
            dataset['negative'][slide_name]['normal'] = np.random.choice(normal, size=400)
        else:
            if not slide_name in not_exhaustively_annotated:
                dataset[diag][slide_name]['normal'] = list(np.random.choice(normal, size=400))

            if not slide_name in not_all_tumor:
                n = len(tumor)
                if n == 0:
                    pass
                elif diag == 'itc':
                    dataset[diag][slide_name]['tumor'] = list(np.random.choice(tumor, size=100))
                    dataset[diag][slide_name]['tumor'] = list(np.random.choice(tumor, size=min(10 * n, 20)))
                elif diag == 'Micro':
                    if len(tumor) < 100:
                        dataset[diag][slide_name]['tumor'] = list(
                            np.random.choice(tumor, size=max(100, 10 * len(tumor))))
                    else:
                        dataset[diag][slide_name]['tumor'] = list(np.random.choice(tumor, size=1000))
                    dataset[diag][slide_name]['tumor'] = list(np.random.choice(tumor, size=min(10 * n, 1000)))
                elif diag == 'Macro':
                    dataset[diag][slide_name]['tumor'] = list(np.random.choice(tumor, size=1500))
                    dataset[diag][slide_name]['tumor'] = list(np.random.choice(tumor, size=min(10 * n, 2000)))
                elif diag == 'Deka Macro':
                    dataset[diag][slide_name]['tumor'] = list(np.random.choice(tumor, size=1800))
                    dataset[diag][slide_name]['tumor'] = list(np.random.choice(tumor, size=1000))
                else:
                    raise ValueError()

    train_data = []
    for diag in dataset:
        print(diag)
        for x in sorted(dataset[diag]):
            for k, v in dataset[diag][x].items():
                print(' ', x, (k, len(v)))
                train_data.extend(v)

    print(Counter([x.split('_')[-1] for x in train_data]))
    np.save('/data/unagi0/fukuta/CAMELYON17/training/train_512_0402.npy', train_data)


def extract_testing_dataset_17(ans_dict_17):
    base_dir = '/data/unagi0/fukuta/CAMELYON17/'
    valid = np.load(os.path.join(base_dir, 'training/test_17.npy'))

    dataset = defaultdict(dict)
    for x in valid:
        slide_name = x.split('/')[-1].split('.')[0]
        patient_name = '_'.join(slide_name.split('_')[:2])
        print(slide_name)
        diag = ans_dict_17[slide_name + '.tif']
        print(diag)
        dataset[diag][slide_name] = {}

        images = np.load(os.path.join(base_dir, 'Valid_Patch_Final_512', patient_name, slide_name + '.npy'))

        if diag == 'negative':
            dataset['negative'][slide_name]['normal'] = [slide_name + '_' + p + '_normal' for p in
                                                         np.random.choice(images, size=500, replace=False)]
        else:
            normal = [slide_name + '_' + image for image in images if image.split('_')[-1] != 'tumor']
            tumor = [slide_name + '_' + image for image in images if image.split('_')[-1] == 'tumor']

            dataset[diag][slide_name]['normal'] = [p for p in np.random.choice(normal, size=500, replace=False)]

            n = len(tumor)
            if diag == 'itc':
                dataset[diag][slide_name]['tumor'] = tumor
            elif diag == 'micro':
                dataset[diag][slide_name]['tumor'] = tumor
            else:
                dataset[diag][slide_name]['tumor'] = np.random.choice(tumor, size=min(1000, n // 5), replace=False)

    test_data = defaultdict(list)
    for diag in dataset:
        print(diag)
        for x in dataset[diag]:
            print(' ', x, [(k, len(v)) for k, v in dataset[diag][x].items()])
            if 'normal' in dataset[diag][x]:
                test_data['normal_512'].extend(dataset[diag][x]['normal'])
            if 'tumor' in dataset[diag][x]:
                test_data['tumor_512'].extend(dataset[diag][x]['tumor'])

    print([(diag, len(v)) for diag, v in sorted(test_data.items())])

    with open('/data/unagi0/fukuta/CAMELYON17/training/diag_256_512_0402.pkl', 'wb') as f:
        pickle.dump(test_data, f)


def enrich_with_mistake_16():
    base_dir = '/data/unagi0/fukuta/CAMELYON16/'

    with open('slide_diag_ans.pkl', 'rb')as f:
        slide_ans_16 = pickle.load(f)

    train_data = np.load('/data/unagi0/fukuta/CAMELYON17/training/train_0329_additional.npy')
    print(Counter([x.split('_')[-1] for x in train_data]))

    train_dataset = defaultdict(list)
    for x in train_data:
        train_dataset['_'.join(x.split('_')[:2])].append(x)

    dataset = {}
    for x in train_dataset:
        dataset[x] = {}
        normal = [l for l in train_dataset[x] if l.split('_')[-1] == 'normal']
        tumor = [l for l in train_dataset[x] if l.split('_')[-1] == 'tumor']
        if len(normal) > 0:
            dataset[x]['normal'] = normal
        if len(tumor) > 0:
            dataset[x]['tumor'] = tumor

    heatmap_dir = os.path.join(base_dir, 'heatmap_256')
    heatmap = {x: sorted(os.listdir(os.path.join(heatmap_dir, x))) for x in os.listdir(heatmap_dir) if
               len(os.listdir(os.path.join(heatmap_dir, x))) > 0}

    result = {}

    big_macro = ['Test_{:03d}'.format(i) for i in [16, 21, 26, 40, 51, 71, 73, 90, 94, 105, 113]]
    for slide_name in sorted(heatmap):
        if slide_name in big_macro:
            continue
        print(slide_name, '-', slide_ans_16[slide_name], end=' - ')

        pred, tumor = heatmap[slide_name]
        pred_map = np.load(os.path.join(heatmap_dir, slide_name, pred))

        valid = np.load(os.path.join('/data/unagi0/fukuta/CAMELYON16/valid_patch_256/', slide_name + '.npy'))
        easy_negative = []
        easy_positive = []
        hard_negative = []
        hard_positive = []
        for path in valid:
            i, j = [int(x) for x in path.split('.')[0].split('_')[:3:2]]
            if path.split('_')[-1] == 'normal':
                if pred_map[i][j] > 0.75:
                    hard_negative.append(slide_name + '_' + path)
                elif pred_map[i][j] < 0.01:
                    easy_negative.append(slide_name + '_' + path)
            elif path.split('_')[-1] == 'tumor':
                if pred_map[i][j] < 0.8:
                    hard_positive.append(slide_name + '_' + path)
                elif pred_map[i][j] > 0.975:
                    easy_positive.append(slide_name + '_' + path)
            else:
                pass
        result[slide_name] = (easy_negative, easy_positive, hard_negative, hard_positive)

    hard_negative_images = []
    hard_positive_images = []
    for x in sorted(result):
        print(x, len(result[x][0]), len(result[x][1]), len(result[x][2]), len(result[x][3]))
        if len(result[x][2]) < 1000:
            hard_negative_images.extend(result[x][2])
        else:
            hard_negative_images.extend(list(np.random.choice(result[x][2], size=1000)))
        if len(result[x][3]) < 1000:
            hard_positive_images.extend(result[x][3])
        else:
            hard_positive_images.extend(list(np.random.choice(result[x][3], size=1000)))

    np.save('hard_samples_16_512', hard_negative_images + hard_positive_images)

    for slide in dataset:
        easy_negative, easy_positive, hard_negative, hard_positive = result[slide]

        tif = tif_load(slide)
        if len(hard_negative) < 64:
            wrong_imgs = hard_negative
        else:
            wrong_imgs = np.random.choice(hard_negative, size=64, replace=False)

        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        count = 0
        for wrong in wrong_imgs:
            count += 1
            l, u = [int(x) for x in wrong.split('_')[6:8]]
            region = tif.read_region((l, u), 0, (256, 256))
            ax = fig.add_subplot(8, 8, count, xticks=[], yticks=[])
            ax.imshow(region)
        plt.show()

    train_new = []
    for x in dataset:
        easy_negative, easy_positive, hard_negative, hard_positive = result[x]
        if 'normal' in dataset[x]:
            v = dataset[x]['normal'].copy()
            if len(hard_negative) > 300:
                v += list(np.random.choice(hard_negative, size=300))
            else:
                v += hard_negative
            en = [x for x in v if x in easy_negative]
            en = set(np.random.choice(en, size=len(en) // 2))

            v_ = list(set(v) - set(en))
            if len(v_) < 100:
                train_new.extend(list(np.random.choice(v, size=100)))
            elif len(v_) < 600:
                train_new.extend(v_)
            else:
                train_new.extend(list(np.random.choice(v_, size=600, replace=False)))

        if 'tumor' in dataset[x]:
            v = dataset[x]['tumor'].copy()
            if slide_ans_16[x] == 'Deka Macro':
                n = 1000
            elif slide_ans_16[x] == 'Micro' or slide_ans_16[x] == 'Macro':
                n = min(int(1.5 * len(v)), 1000)
            else:
                n = len(v)
            print(x, slide_ans_16[x], n)
            if len(hard_positive) > 300:
                v = set(v) | set(np.random.choice(hard_positive, size=300))
            else:
                v = set(v) | set(hard_positive)

            v = list(np.random.choice(list(v), size=n))
            train_new.extend(v)

    for x in sorted(result):
        print(x, len(result[x]))

    additional_data = []
    noneed = []
    for x in sorted(result):
        if len(result[x]) <= 1:
            noneed.append(x)
            continue
        if x in big_macro:
            continue

        print(x, slide_ans_16[x], len(result[x]))

        res = result[x]
        n = len(res)
        if slide_ans_16[x] == 'None':
            if n == 0:
                pass
            elif n < 100:
                additional_data.extend(list(np.random.choice(res, size=100)))
            else:
                additional_data.extend(res)
        else:
            normal = [t for t in res if t.split('_')[-1] == 'normal']
            tumor = [t for t in res if t.split('_')[-1] == 'tumor']
            if len(normal) == 0:
                pass
            elif len(normal) < 100:
                additional_data.extend(list(np.random.choice(normal, size=100)))
            else:
                additional_data.extend(res)

            if len(tumor) == 0:
                pass
            elif len(tumor) < 100:
                additional_data.extend(list(np.random.choice(tumor, size=len(tumor) * 5)))
            elif len(tumor) < 500:
                additional_data.extend(list(np.random.choice(tumor, size=500)))
            else:
                additional_data.extend(res)

    print(Counter([x.split('_')[-1] for x in additional_data]))
    train_additional = [x for x in train_data if '_'.join(x.split('_')[:2]) not in noneed]
    print(Counter([x.split('_')[-1] for x in train_additional]))
    train_additional += additional_data
    print(Counter([x.split('_')[-1] for x in train_additional]))

    np.save('/data/unagi0/fukuta/CAMELYON17/training/train_diag_0326_additional', train_additional)


def enrich_with_mistake_17():
    base_dir = '/data/unagi0/fukuta/CAMELYON17/'

    csv = pd.read_csv(os.path.join(base_dir, 'training/stage_labels.csv'))
    patient_ans = {a.split('.')[0]: b for a, b in csv.ix[np.arange(100) * 6].values}

    slide_ans_17 = {}
    for a, b in csv.values:
        if a.endswith('.tif'):
            slide_ans_17[a] = b

    train_data = np.load('/data/unagi0/fukuta/CAMELYON17/training/train_512_0330.npy')
    print(Counter([x.split('_')[-1] for x in train_data]))

    train_dataset = defaultdict(list)
    for x in train_data:
        train_dataset['_'.join(x.split('_')[:2])].append(x)

    dataset = {}
    for x in train_dataset:
        dataset[x] = {}
        normal = [l for l in train_dataset[x] if l.split('_')[-1] == 'normal']
        tumor = [l for l in train_dataset[x] if l.split('_')[-1] == 'tumor']
        if len(normal) > 0:
            dataset[x]['normal'] = normal
        if len(tumor) > 0:
            dataset[x]['tumor'] = tumor

    heatmap_dir = os.path.join(base_dir, 'heatmap_512_texture')
    heatmap = {x: sorted(os.listdir(os.path.join(heatmap_dir, x))) for x in os.listdir(heatmap_dir) if
               len(os.listdir(os.path.join(heatmap_dir, x))) > 0}

    result = {}

    for slide_name in sorted(heatmap):
        patient = '_'.join(slide_name.split('_')[:2])
        if int(patient.split('_')[1]) >= 100:
            continue

        print(slide_name, '-', slide_ans_17[slide_name + '.tif'], end=' - ')

        pred, tumor = heatmap[slide_name]
        pred_map = np.load(os.path.join(heatmap_dir, slide_name, pred))

        valid = np.load(
            os.path.join('/data/unagi0/fukuta/CAMELYON17/Valid_Patch_New_512/', patient, slide_name + '.npy'))
        easy_negative = []
        easy_positive = []
        hard_negative = []
        hard_positive = []
        for path in valid:
            i, j = [int(x) for x in path.split('.')[0].split('_')[:3:2]]
            if path.split('_')[-1] == 'normal':
                if pred_map[i][j] > 0.8:
                    hard_negative.append(slide_name + '_' + path)
                elif pred_map[i][j] < 0.01:
                    easy_negative.append(slide_name + '_' + path)
            elif slide_ans_17[slide_name + '.tif'] == 'negative':
                if pred_map[i][j] > 0.8:
                    hard_negative.append(slide_name + '_' + path + '_normal')
                elif pred_map[i][j] < 0.01:
                    easy_negative.append(slide_name + '_' + path + '_normal')
            elif path.split('_')[-1] == 'tumor':
                if pred_map[i][j] < 0.8:
                    hard_positive.append(slide_name + '_' + path)
                elif pred_map[i][j] > 0.975:
                    easy_positive.append(slide_name + '_' + path)
            else:
                pass
        result[slide_name] = (easy_negative, easy_positive, hard_negative, hard_positive)

    hard_negative_images = []
    hard_positive_images = []
    for x in sorted(result):
        print(x, len(result[x][0]), len(result[x][1]), len(result[x][2]), len(result[x][3]))
        if len(result[x][2]) < 1000:
            hard_negative_images.extend(result[x][2])
        else:
            hard_negative_images.extend(list(np.random.choice(result[x][2], size=1000)))
        if len(result[x][3]) < 1000:
            hard_positive_images.extend(result[x][3])
        else:
            hard_positive_images.extend(list(np.random.choice(result[x][3], size=1000)))

    np.save('hard_samples_17_512', hard_negative_images + hard_positive_images)

    train_new = []
    for diag in dataset:
        print(diag)
        for x in sorted(dataset[diag]):
            easy_negative, easy_positive, hard_negative, hard_positive = result[x]
            if 'normal' in dataset[diag][x]:
                v = dataset[diag][x]['normal'].copy()
                if len(hard_negative) > 300:
                    v += list(np.random.choice(hard_negative, size=300))
                else:
                    v += hard_negative
                en = [x for x in v if x in easy_negative]
                en = set(np.random.choice(en, size=len(en) // 2))

                v_ = list(set(v) - set(en))
                if len(v_) < 300:
                    train_new.extend(list(np.random.choice(v, size=300)))
                elif len(v_) < 500:
                    train_new.extend(v_)
                else:
                    train_new.extend(list(np.random.choice(v_, size=500, replace=False)))

                    #             print(len(v), len([x for x in v if x in easy_negative]), len(hard_negative))
            if 'tumor' in dataset[diag][x]:
                v = dataset[diag][x]['tumor'].copy()
                n = len(v)
                if len(hard_positive) > 300:
                    v = set(v) | set(np.random.choice(hard_positive, size=300))
                else:
                    v = set(v) | set(hard_positive)

                v = list(np.random.choice(list(v), size=n))
                train_new.extend(v)
