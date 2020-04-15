import os
from PIL import Image

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

@register_dataset_obj('MOZ_S2')
class MOZ_S2(MOZ):

    name = 'MOZ_S2'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(MOZ_S2, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                     split=split, transform=transform)
        if self.dset == 'val':
            self.dset = 'test'  # MOZ does not use val for now.
        ann_dir = os.path.join(self.ann_root, '{}_mix_season_2.txt'.format(self.dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()


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
