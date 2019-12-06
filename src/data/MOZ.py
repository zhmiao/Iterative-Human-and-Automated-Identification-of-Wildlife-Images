import os

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
                assert line_sp[1] in self.class_labels
                self.data.append(line_sp[0])
                self.labels.append(self.class_indices[line_sp[1]])


@register_dataset_obj('MOZ_S1')
class MOZ_S1(MOZ):

    name = 'MOZ_S1'

    def __init__(self, rootdir, class_indices, dset='train', transform=None):
        super(MOZ_S1, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset, transform=transform)
        if self.dset == 'val':
            self.dset = 'test'  # MOZ does not use val for now.
        ann_dir = os.path.join(self.ann_root, '{}_season_{}.txt'.format(self.dset, 1))
        self.load_data(ann_dir)


@register_dataset_obj('MOZ_S2')
class MOZ_S2(MOZ):

    name = 'MOZ_S2'

    def __init__(self, rootdir, class_indices, dset='train', transform=None):
        super(MOZ_S2, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset, transform=transform)
        if self.dset == 'val':
            self.dset = 'test'  # MOZ does not use val for now.
        ann_dir = os.path.join(self.ann_root, '{}_season_{}.txt'.format(self.dset, 2))
        self.load_data(ann_dir)

