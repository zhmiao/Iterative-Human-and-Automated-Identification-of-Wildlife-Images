import os
import random
from PIL import Image, ImageFilter
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from .utils import register_dataset_obj, BaseDataset


class MOZ(BaseDataset):

    def __init__(self, rootdir, class_indices, dset='train', transform=None):
        super(MOZ, self).__init__(class_indices=class_indices, dset=dset, transform=transform)
        self.img_root = os.path.join(rootdir, 'Mozambique')
        self.ann_root = os.path.join(rootdir, 'Mozambique', 'SplitLists')

    def load_data(self, ann_dir):
        with open(ann_dir, 'r') as f:
            for line in f:
                line_sp = line.replace('\n', '').split(' ')

                if line_sp[1] in self.class_indices.keys():
                    label = self.class_indices[line_sp[1]]
                else:
                    label = -1

                self.data.append(line_sp[0])
                self.labels.append(label)


@register_dataset_obj('MOZ_S1_LT')
class MOZ_S1_LT(MOZ):

    name = 'MOZ_S1_LT'

    def __init__(self, rootdir, class_indices, dset='train', transform=None):
        super(MOZ_S1_LT, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                        transform=transform)
        ann_dir = os.path.join(self.ann_root, '{}_mix_season_1_lt.txt'.format(self.dset))
        self.load_data(ann_dir)


@register_dataset_obj('MOZ_S2_LT_FULL')
class MOZ_S2_LT_FULL(MOZ):

    name = 'MOZ_S2_LT_FULL'

    def __init__(self, rootdir, class_indices, dset='train', transform=None):
        super(MOZ_S2_LT_FULL, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                             transform=transform)
        ann_dir = os.path.join(self.ann_root, '{}_mix_season_2_lt.txt'.format(self.dset))
        self.load_data(ann_dir)


@register_dataset_obj('MOZ_UNKNOWN')
class MOZ_MIX_OOD(MOZ):

    name = 'MOZ_UNKNOWN'

    def __init__(self, rootdir, class_indices, dset='train', transform=None):
        super(MOZ_MIX_OOD, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                          transform=transform)
        ann_dir = os.path.join(self.ann_root, '{}_mix_ood.txt'.format(dset))
        self.load_data(ann_dir)


@register_dataset_obj('MOZ_S3_ALL')
class MOZ_S3_ALL(Dataset):

    def __init__(self, rootdir, class_indices, dset=None, transform=None):
        self.img_root = os.path.join(rootdir, 'Mozambique', 'Mozambique_season_3')
        self.ann_root = os.path.join(rootdir, 'Mozambique', 'SplitLists')
        self.class_indices = class_indices
        self.transform = transform
        self.data = []
        ann_dir = os.path.join(self.ann_root, 'Mozambique_season_3_all.txt')
        self.load_data(ann_dir)

    def load_data(self, ann_dir):
        with open(ann_dir, 'r') as f:
            for line in f:
                line_sp = line.replace('\n', '')
                self.data.append(line_sp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_id = self.data[index]
        file_dir = os.path.join(self.img_root, file_id)
        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, file_id


@register_dataset_obj('MOZ_S2_GTPS')
class MOZ_S2_GTPS(MOZ):

    name = 'MOZ_S2_GTPS'

    def __init__(self, rootdir, class_indices, dset='train', transform=None,
                 conf_preds=None, pseudo_labels_hard=None, pseudo_labels_soft=None,
                 GTPS_mode='GT', blur=False):

        super(MOZ_S2_GTPS, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset, 
                                          transform=transform)

        ann_dir = os.path.join(self.ann_root, '{}_mix_season_2_lt.txt'.format(self.dset))
        self.load_data(ann_dir)

        self.conf_preds = conf_preds
        self.pseudo_labels_hard = pseudo_labels_hard
        self.pseudo_labels_soft = pseudo_labels_soft
        self.unconf_only = unconf_only
        self.blur = blur

        if self.blur:
            print('** USING BLURING **')

        if self.conf_preds is not None:
            print('Confidence prediction is not NONE.\n')

        if GTPS_mode == 'both':
            print('** LOADING BOTH GROUND TRUTH AND PSEUDO LABELS **')
            assert pseudo_labels_hard is not None and pseudo_labels_soft is not None
            self.pseudo_label_selection()
        elif GTPS_mode == 'GT':
            print('** LOADING ONLY GROUND TRUTH **')
            self.pick_unconf()
        elif GTPS_mode == 'PS':
            print('** LOADING ONLY PSEUDO LABELS **')
            assert pseudo_labels_hard is not None and pseudo_labels_soft is not None
            self.pseudo_label_selection()
            self.pick_conf()
        elif GTPS_mode is None:
            print('** NOT USING GTPS MODES **')

    def class_counts_cal_ann(self):

        if self.unconf_only:
            _, ann_counts = self.class_counts_cal()
            return ann_counts

        else:
            unique_labels = np.unique(self.labels)
            unique_ann, unique_ann_counts = np.unique(np.array(self.labels)[np.array(self.conf_preds) == 0],
                                                      return_counts=True)
            temp_dic = {l: c for l, c in zip(unique_ann, unique_ann_counts)}
            ann_counts = np.array([0 for _ in range(len(unique_labels))])
            for l in unique_labels:
                if l in temp_dic:
                    ann_counts[l] = temp_dic[l]
            return ann_counts

    def pick_unconf(self):
        print('** PICKING GROUND TRUTHED DATA **')
        data = np.array(self.data)
        labels = np.array(self.labels)
        conf_preds = np.array(self.conf_preds)
        self.data = list(data[conf_preds == 0])
        self.labels = list(labels[conf_preds == 0])

    def pick_conf(self):
        print('** PICKING PSEUDO LABLED DATA **')
        data = np.array(self.data)
        labels = np.array(self.labels)
        conf_preds = np.array(self.conf_preds)
        self.data = list(data[conf_preds == 1])
        self.labels = list(labels[conf_preds == 1])

    def pseudo_label_accuracy(self):
        pseudo_labels = np.array(self.pseudo_labels)
        labels = np.array(self.labels)
        print('** CHECKING PSEUDO LABEL ACCURACY **')
        conf_pseudo_labels = pseudo_labels[pseudo_labels != -1]
        conf_labels = labels[pseudo_labels != -1]
        acc = ((conf_pseudo_labels == conf_labels).sum() / len(conf_labels)).mean()
        print('PSEUDO LABEL ACCURACY: {:3f}'.format(acc * 100))

    def pseudo_label_selection(self):

        conf_preds = np.array(self.conf_preds)
        pseudo_labels = np.array(self.pseudo_labels)
        labels = np.array(self.labels)
        data = np.array(self.data)

        self.pseudo_label_accuracy()

        print('** INFUSING PSEUDO LABELS AND SELECTING DATA **')

        labels[conf_preds == 1] = pseudo_labels[conf_preds == 1]

        # Pick data first then labels
        data = data[labels != -1]
        labels = labels[labels != -1]

        self.labels = list(labels)
        self.data = list(data)

    def __getitem__(self, index):

        file_id = self.data[index]
        label = self.labels[index]
        file_dir = os.path.join(self.img_root, file_id)

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')
            if self.blur:
                blur_r = random.randint(0, 12) / 10
                sample = sample.filter(ImageFilter.GaussianBlur(radius=blur_r))

        if self.transform is not None:
            sample = self.transform(sample)

        if self.conf_preds is not None:
            conf_pred = self.conf_preds[index]
            return sample, label, conf_pred, index
        else:
            return sample, label



class MOZ_ST2_SoftIter(MOZ):

    def __init__(self, rootdir, class_indices, dset='train', 
                 transform=None, conf_preds=None, pseudo_labels_hard=None, pseudo_labels_soft=None,
                 unconf_only=False, blur=False):

        super(MOZ_ST2_SoftIter, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset, 
                                               transform=transform)
        self.conf_preds = conf_preds
        self.pseudo_labels_hard = pseudo_labels_hard
        self.pseudo_labels_soft = pseudo_labels_soft
        self.unconf_only = unconf_only
        self.blur = blur
        if self.blur:
            print('** USING BLURING **')
        if self.conf_preds is not None:
            print('Confidence prediction is not NONE.\n')

    def class_counts_cal_ann(self):

        if self.unconf_only:
            _, ann_counts = self.class_counts_cal()
            return ann_counts

        else:
            unique_labels = np.unique(self.labels)
            unique_ann, unique_ann_counts = np.unique(np.array(self.labels)[np.array(self.conf_preds) == 0],
                                                      return_counts=True)
            temp_dic = {l: c for l, c in zip(unique_ann, unique_ann_counts)}
            ann_counts = np.array([0 for _ in range(len(unique_labels))])
            for l in unique_labels:
                if l in temp_dic:
                    ann_counts[l] = temp_dic[l]
            return ann_counts

    def pick_unconf(self):
        print('** PICKING GROUND TRUTHED DATA AND SOFT LABELS**')
        data = np.array(self.data)
        labels = np.array(self.labels)
        soft = np.array(self.pseudo_labels_soft)
        conf_preds = np.array(self.conf_preds)
        self.data = list(data[conf_preds == 0])
        self.labels = list(labels[conf_preds == 0])
        self.pseudo_labels_soft = [list(l) for l in soft[conf_preds == 0]]

    def pick_unconf_no_soft(self):
        print('** PICKING GROUND TRUTHED DATA**')
        data = np.array(self.data)
        labels = np.array(self.labels)
        conf_preds = np.array(self.conf_preds)
        self.data = list(data[conf_preds == 0])
        self.labels = list(labels[conf_preds == 0])

    def pick_conf(self):
        print('** PICKING PSEUDO LABLED DATA HARD AND SOFT**')
        data = np.array(self.data)
        labels = np.array(self.labels)
        soft = np.array(self.pseudo_labels_soft)
        conf_preds = np.array(self.conf_preds)
        self.data = list(data[conf_preds == 1])
        self.labels = list(labels[conf_preds == 1])
        self.pseudo_labels_soft = [list(l) for l in soft[conf_preds == 1]]

    def pseudo_label_accuracy(self):
        pseudo_labels_hard = np.array(self.pseudo_labels_hard)
        labels = np.array(self.labels)
        print('** CHECKING HARD PSEUDO LABEL ACCURACY **')
        conf_pseudo_labels_hard = pseudo_labels_hard[pseudo_labels_hard != -1]
        conf_labels = labels[pseudo_labels_hard != -1]
        acc = ((conf_pseudo_labels_hard == conf_labels).sum() / len(conf_labels)).mean()
        print('HARD PSEUDO LABEL ACCURACY: {:3f}'.format(acc * 100))

    def pseudo_label_selection(self):

        conf_preds = np.array(self.conf_preds)
        pseudo_labels_hard = np.array(self.pseudo_labels_hard)
        pseudo_labels_soft = np.array(self.pseudo_labels_soft)
        labels = np.array(self.labels)
        data = np.array(self.data)

        self.pseudo_label_accuracy()

        print('** INFUSING PSEUDO LABELS AND SELECTING DATA **')

        labels[conf_preds == 1] = pseudo_labels_hard[conf_preds == 1]

        # Pick data and soft labels first then labels
        data = data[labels != -1]
        pseudo_labels_soft = pseudo_labels_soft[labels != -1]
        labels = labels[labels != -1]

        self.labels = list(labels)
        self.data = list(data)
        self.pseudo_labels_soft = [list(l) for l in pseudo_labels_soft]

    def __getitem__(self, index):

        file_id = self.data[index]
        label = self.labels[index]

        file_dir = os.path.join(self.img_root, file_id)

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')
            if self.blur:
                blur_r = random.randint(0, 12) / 10
                sample = sample.filter(ImageFilter.GaussianBlur(radius=blur_r))

        if self.transform is not None:
            sample = self.transform(sample)

        if self.conf_preds is not None:
            if self.pseudo_labels_soft is not None:
                soft_label = self.pseudo_labels_soft[index]
                conf_pred = self.conf_preds[index]
                return sample, label, torch.tensor(soft_label), conf_pred, index
            else:
                conf_pred = self.conf_preds[index]
                return sample, label, conf_pred, index
        else:
            return sample, label


@register_dataset_obj('MOZ_S2_GTPS_SoftIter')
class MOZ_S2_GTPS_SoftIter(MOZ_ST2_SoftIter):

    name = 'MOZ_S2_GTPS_SoftIter'

    def __init__(self, rootdir, class_indices, dset='train', 
                 transform=None, conf_preds=None, pseudo_labels_hard=None, pseudo_labels_soft=None,
                 GTPS_mode='both', blur=False):
        super(MOZ_S2_GTPS_SoftIter, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                                   transform=transform, conf_preds=conf_preds,
                                                   pseudo_labels_hard=pseudo_labels_hard,
                                                   pseudo_labels_soft=pseudo_labels_soft,
                                                   blur=blur)

        ann_dir = os.path.join(self.ann_root, '{}_mix_season_2.txt'.format(self.dset))

        self.load_data(ann_dir)

        if GTPS_mode == 'both':
            print('** LOADING BOTH GROUND TRUTH AND PSEUDO LABELS (HARD AND SOFT) **')
            assert pseudo_labels_hard is not None and pseudo_labels_soft is not None
            self.pseudo_label_selection()
        elif GTPS_mode == 'GT':
            print('** LOADING ONLY GROUND TRUTH AND SOFT LABELS**')
            self.pick_unconf()
        elif GTPS_mode == 'GT_ONLY':
            print('** LOADING ONLY GROUND TRUTH LABELS**')
            self.pick_unconf_no_soft()
        elif GTPS_mode == 'PS':
            print('** LOADING ONLY PSEUDO LABELS (HARD AND SOFT)**')
            assert pseudo_labels_hard is not None and pseudo_labels_soft is not None
            self.pseudo_label_selection()
            self.pick_conf()
        elif GTPS_mode is None:
            print('** NOT USING GTPS MODES **')

