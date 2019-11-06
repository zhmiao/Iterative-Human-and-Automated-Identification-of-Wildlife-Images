import os
import json
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from .utils import register_dataset_obj

class CCT(Dataset):

    def __init__(self, rootdir, dset='train', transform=None):
        self.img_root = os.path.join(rootdir, 'CCT_15', 'eccv_18_all_images_256')
        self.ann_root = os.path.join(rootdir, 'CCT_15', 'eccv_18_annotation_files')
        self.dset = dset
        self.transform = transform
        self.data = None
        self.categories_names = {}
        self.categories_labels = {}

    def load_json(self, json_dir):
        with open(json_dir, 'r') as js:
            ann_js = json.load(js)

        self.categories_names = ann_js['categories']
        assert len(self.categories_names) == 16, 'Class label problems. \n'
        self.data = [entry for entry in ann_js['annotations'] if entry['category_id'] != 30]

        label = 0
        for cat in self.categories_names:
            if cat['id'] != 30:
                self.categories_labels[cat['id']] = label
                label += 1

    def class_num_cal(self):
        labels = []
        for entry in self.data:
            labels.append(self.categories_labels[entry['category_id']])
        return np.unique(labels, return_counts=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_id = self.data[index]['image_id']
        label = self.categories_labels[self.data[index]['category_id']]
        file_dir = os.path.join(self.img_root, file_id + '.jpg')

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


@register_dataset_obj('CCT_cis')
class CCT_cis(CCT):

    name = 'CCT_cis'

    def __init__(self, rootdir, dset='train', transform=None):
        super(CCT_cis, self).__init__(rootdir=rootdir, dset=dset, transform=transform)
        json_dir = os.path.join(self.ann_root, 'cis_{}_annotations.json'.format(dset))
        self.load_json(json_dir)


@register_dataset_obj('CCT_trans')
class CCT_trans(CCT):

    name = 'CCT_trans'

    def __init__(self, rootdir, dset='test', transform=None):
        super(CCT_trans, self).__init__(rootdir=rootdir, dset=dset, transform=transform)
        assert self.dset != 'train', 'CCT_trans does not have training data currently. \n'
        json_dir = os.path.join(self.ann_root, 'trans_{}_annotations.json'.format(dset))
        self.load_json(json_dir)



