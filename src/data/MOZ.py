import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from .utils import register_dataset_obj, BaseDataset


class MOZ(BaseDataset):

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(MOZ, self).__init__(class_indices=class_indices, dset=dset, split=split, transform=transform)
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


@register_dataset_obj('MOZ_S1')
class MOZ_S1(MOZ):

    name = 'MOZ_S1'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(MOZ_S1, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                     split=split, transform=transform)
        if self.dset == 'val':
            self.dset = 'test'  # MOZ does not use val for now.
        ann_dir = os.path.join(self.ann_root, '{}_mix_season_1.txt'.format(self.dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()



@register_dataset_obj('MOZ_EP')
class MOZ_EP(MOZ):

    name = 'MOZ_EP'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(MOZ_EP, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                     split=split, transform=transform)
        if self.dset == 'val':
            self.dset = 'test'  # MOZ does not use val for now.
        ann_dir = os.path.join(self.ann_root, '{}_empty_picker.txt'.format(self.dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()

@register_dataset_obj('MOZ_S3_ALL')
class MOZ_S3_ALL(Dataset):

    def __init__(self, rootdir, class_indices, dset=None, split=None, transform=None):
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


@register_dataset_obj('MOZ_S3_NEP')
class MOZ_S3_NEP(MOZ_S3_ALL):

    def __init__(self, rootdir, class_indices, dset=None, split=None, transform=None):
        self.img_root = os.path.join(rootdir, 'Mozambique', 'Mozambique_season_3')
        self.ann_root = os.path.join(rootdir, 'Mozambique', 'SplitLists')
        self.class_indices = class_indices
        self.transform = transform
        self.data = []
        ann_dir = os.path.join(self.ann_root, 'Mozambique_season_3_NEP.txt')
        self.load_data(ann_dir)


@register_dataset_obj('MOZ_S1_10')
class MOZ_S1_10(MOZ):

    name = 'MOZ_S1_10'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(MOZ_S1_10, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                        split=split, transform=transform)
        if self.dset == 'val':
            self.dset = 'test'  # MOZ does not use val for now.
        # ann_dir = os.path.join(self.ann_root, '{}_mix_10_season_1.txt'.format(self.dset))
        ann_dir = os.path.join(self.ann_root, '{}_mix_season_1.txt'.format(self.dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()


class MOZ_ORI(BaseDataset):

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(MOZ_ORI, self).__init__(class_indices=class_indices, dset=dset, split=split, transform=transform)
        self.img_root = os.path.join(rootdir, 'Mozambique', 'MOZ-30')
        self.ann_root = os.path.join(rootdir, 'Mozambique', 'MOZ-30', 'annotations')

    def load_data(self, ann_dir):
        with open(ann_dir, 'r') as f:
            for line in f:
                line_sp = line.replace('\n', '').split(' ')
                self.data.append(line_sp[0].replace('/home/zhmiao/datasets/ecology/Mozambique/', ''))
                self.labels.append(int(line_sp[1]))

    def __getitem__(self, index):
        file_id = self.data[index]
        label = self.labels[index]
        file_dir = os.path.join(self.img_root, file_id)
        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


@register_dataset_obj('MOZ_S1_ORI')
class MOZ_S1_ORI(MOZ_ORI):

    name = 'MOZ_S1_ORI'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(MOZ_S1_ORI, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                     split=split, transform=transform)
        if self.dset == 'val':
            self.dset = 'test'  # MOZ does not use val for now.
        ann_dir = os.path.join(self.ann_root, '{}20_sp.txt'.format(self.dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()


class MOZ_ST2(MOZ):

    def __init__(self, rootdir, class_indices, dset='train', split=None,
                 transform=None, conf_preds=None, unconf_only=False):

        super(MOZ_ST2, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset, split=split,
                                      transform=transform)
        self.conf_preds = conf_preds
        self.unconf_only = unconf_only
        if self.conf_preds is not None:
            print('Confidence prediction is not NONE.\n')

    def class_counts_cal_ann(self):
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
        print('** PICKING UNCONFIDENT DATA ONLY **')
        data = np.array(self.data)
        labels = np.array(self.labels)
        conf_preds = np.array(self.conf_preds)
        self.data = list(data[conf_preds == 0])
        self.labels = list(labels[conf_preds == 0])

    def __getitem__(self, index):

        file_id = self.data[index]
        label = self.labels[index]
        file_dir = os.path.join(self.img_root, file_id)

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.conf_preds is not None:
            conf_pred = self.conf_preds[index]
            return sample, label, conf_pred, index
        else:
            return sample, label


@register_dataset_obj('MOZ_S2')
class MOZ_S2(MOZ_ST2):

    name = 'MOZ_S2'

    def __init__(self, rootdir, class_indices, dset='train', split=None,
                 transform=None, conf_preds=None, unconf_only=False):
        super(MOZ_S2, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                     split=split, transform=transform, conf_preds=conf_preds,
                                     unconf_only=unconf_only)
        if self.dset == 'val':
            self.dset = 'test'  # MOZ does not use val for now.
        ann_dir = os.path.join(self.ann_root, '{}_mix_season_2.txt'.format(self.dset))
        self.load_data(ann_dir)
        if unconf_only:
            self.pick_unconf()
        if split is not None:
            self.data_split()


@register_dataset_obj('MOZ_S2_LEFTOUT')
class MOZ_S2_LEFTOUT(MOZ):

    name = 'MOZ_S2_LEFTOUT'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(MOZ_S2_LEFTOUT, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                             split=split, transform=transform)
        ann_dir = os.path.join(self.ann_root, 'test_mix_season_2_with_leftout.txt')
        self.load_data(ann_dir)






















