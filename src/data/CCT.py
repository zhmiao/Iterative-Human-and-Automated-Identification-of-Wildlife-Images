import os
import json
from PIL import Image, ImageOps

from .utils import register_dataset_obj, BaseDataset


class CCT(BaseDataset):

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(CCT, self).__init__(class_indices=class_indices, dset=dset, split=split, transform=transform)
        self.img_root = os.path.join(rootdir, 'CCT_15', 'eccv_18_all_images_256')
        self.ann_root = os.path.join(rootdir, 'CCT_15', 'eccv_18_annotation_files')

    def load_data(self, ann_dir):
        with open(ann_dir, 'r') as js:
            ann_js = json.load(js)

        annotations = [entry
                       for entry in ann_js['annotations']
                       if entry['category_id'] != 30
                       and entry['category_id'] != 33]

        for entry in annotations:
            self.data.append(entry['image_id'])
            assert entry['category_id'] in self.class_indices.keys()
            self.labels.append(self.class_indices[entry['category_id']])


@register_dataset_obj('CCT_CIS_S1')
class CCT_CIS_S1(CCT):

    name = 'CCT_CIS_S1'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(CCT_CIS_S1, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                         split=split, transform=transform)
        ann_dir = os.path.join(self.ann_root, 'cis_{}_annotations_season_1.json'.format(dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()


@register_dataset_obj('CCT_CIS_S2')
class CCT_CIS_S2(CCT):

    name = 'CCT_CIS_S2'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(CCT_CIS_S2, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                         split=split, transform=transform)
        ann_dir = os.path.join(self.ann_root, 'cis_{}_annotations_season_2.json'.format(dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()


@register_dataset_obj('CCT_CIS_ALL')
class CCT_CIS_ALL(CCT):

    name = 'CCT_CIS_ALL'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(CCT_CIS_ALL, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                          split=split, transform=transform)
        ann_dir = os.path.join(self.ann_root, 'cis_{}_annotations.json'.format(dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()


@register_dataset_obj('CCT_TRANS')
class CCT_TRANS(CCT):

    name = 'CCT_TRANS'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(CCT_TRANS, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                        split=split, transform=transform)
        assert self.dset != 'train', 'CCT_TRANS does not have training data currently. \n'
        ann_dir = os.path.join(self.ann_root, 'trans_{}_annotations.json'.format(dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()


class CCT_CROP(BaseDataset):

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(CCT_CROP, self).__init__(class_indices=class_indices, dset=dset, split=split, transform=transform)
        self.img_root = os.path.join(rootdir, 'CCT_15', 'eccv_18_all_images_256')
        self.ann_root = os.path.join(rootdir, 'CCT_15', 'eccv_18_annotation_files')
        self.bbox = []

    def load_data(self, ann_dir):
        with open(ann_dir, 'r') as js:
            ann_js = json.load(js)

        annotations = [entry
                       for entry in ann_js['annotations']
                       if entry['category_id'] != 30
                       and entry['category_id'] != 33]

        for entry in annotations:
            if 'bbox' in entry:
                self.data.append(entry['image_id'])
                assert entry['category_id'] in self.class_indices.keys()
                self.labels.append(self.class_indices[entry['category_id']])
                self.bbox.append(entry['bbox'])

    def __getitem__(self, index):
        file_id = self.data[index]
        label = self.labels[index]
        bbox = self.bbox[index]
        file_dir = os.path.join(self.img_root, file_id)
        if not file_dir.endswith('.JPG'):
            file_dir += '.jpg'

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB').crop(bbox)
            sample = ImageOps.expand(sample, tuple((max(sample.size) - s) // 2 for s in list(sample.size)))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


@register_dataset_obj('CCT_CIS_CROP_S1')
class CCT_CIS_CROP_S1(CCT_CROP):

    name = 'CCT_CIS_CROP_S1'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(CCT_CIS_CROP_S1, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                              split=split, transform=transform)
        ann_dir = os.path.join(self.ann_root, 'cis_{}_annotations_season_1.json'.format(dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()


@register_dataset_obj('CCT_CIS_CROP_S2')
class CCT_CIS_CROP_S2(CCT_CROP):

    name = 'CCT_CIS_CROP_S2'

    def __init__(self, rootdir, class_indices, dset='train', split=None, transform=None):
        super(CCT_CIS_CROP_S2, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                              split=split, transform=transform)
        ann_dir = os.path.join(self.ann_root, 'cis_{}_annotations_season_2.json'.format(dset))
        self.load_data(ann_dir)
        if split is not None:
            self.data_split()

