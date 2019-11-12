import os
import json

from .utils import register_dataset_obj, BaseDataset


class CCT(BaseDataset):

    def __init__(self, rootdir, class_indices, dset='train', transform=None):
        super(CCT, self).__init__(class_indices=class_indices, dset=dset, transform=transform)
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


@register_dataset_obj('CCT_cis')
class CCT_cis(CCT):

    name = 'CCT_cis'

    def __init__(self, rootdir, class_indices, dset='train', transform=None):
        super(CCT_cis, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset, transform=transform)
        ann_dir = os.path.join(self.ann_root, 'cis_{}_annotations.json'.format(dset))
        self.load_data(ann_dir)


@register_dataset_obj('CCT_trans')
class CCT_trans(CCT):

    name = 'CCT_trans'

    def __init__(self, rootdir, class_indices, dset='train', transform=None):
        super(CCT_trans, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset, transform=transform)
        assert self.dset != 'train', 'CCT_trans does not have training data currently. \n'
        ann_dir = os.path.join(self.ann_root, 'trans_{}_annotations.json'.format(dset))
        self.load_data(ann_dir)



