import os
import json
import numpy as np
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
            # Assigne -1 to unseen classes
            if entry['category_id'] in self.class_indices.keys():
                label = self.class_indices[entry['category_id']]
            else:
                label = -1
            self.labels.append(label)


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
        super(CCT_CROP, self).__init__(class_indices=class_indices, dset=dset, split=split,
                                       transform=transform)
        self.img_root = os.path.join(rootdir, 'CCT_15', 'eccv_18_cropped')
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
                # Assigne -1 to unseen classes
                if entry['category_id'] in self.class_indices.keys():
                    label = self.class_indices[entry['category_id']]
                else:
                    label = -1
                self.labels.append(label)
                self.bbox.append(entry['bbox'])

    def data_split(self):
        print('Splitting data to {} samples each class maximum.'.format(self.split))

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.bbox = np.array(self.bbox)

        data_sel = np.empty(shape=0)
        labels_sel = np.empty(shape=0)
        bbox_sel = np.empty(shape=(0, self.bbox.shape[1]))

        unique_labels, unique_counts = np.unique(self.labels, return_counts=True)

        for label, counts in zip(unique_labels, unique_counts):

            data_cat = self.data[self.labels == label]
            labels_cat = self.labels[self.labels == label]
            bbox_cat = self.bbox[self.labels == label]

            if counts > self.split:

                np.random.seed(label)

                indices_sel = np.random.choice(np.arange(len(data_cat)), self.split, replace=False)

                data_sel = np.concatenate((data_sel, data_cat[indices_sel]))
                labels_sel = np.concatenate((labels_sel, labels_cat[indices_sel]))
                bbox_sel = np.concatenate((bbox_sel, bbox_cat[indices_sel]))
            else:
                data_sel = np.concatenate((data_sel, data_cat))
                labels_sel = np.concatenate((labels_sel, labels_cat))
                bbox_sel = np.concatenate((bbox_sel, bbox_cat))

        self.data = list(data_sel)
        self.labels = list(labels_sel.astype(int))
        self.bbox = list(bbox_sel)

    def __getitem__(self, index):
        file_id = self.data[index]
        label = self.labels[index]
        bbox = self.bbox[index]
        pil_bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        file_dir = os.path.join(self.img_root, file_id)
        if not file_dir.endswith('.JPG'):
            file_dir += '.jpg'

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB').crop(pil_bbox)
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


class CCT_CROP_ST2(CCT_CROP):

    def __init__(self, rootdir, class_indices, dset='train', split=None,
                 transform=None, conf_preds=None, unknown_only=False):
        super(CCT_CROP_ST2, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset, split=split,
                                           transform=transform)
        self.conf_preds = conf_preds
        self.unknown_only = unknown_only
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

    def pick_unknown(self):
        print('** PICKING UNKNOWN DATA ONLY **')
        data = np.array(self.data)
        labels = np.array(self.labels)
        bbox = np.array(self.bbox)
        conf_preds = np.array(self.conf_preds)
        self.data = list(data[conf_preds == 0])
        self.labels = list(labels[conf_preds == 0])
        self.bbox = list(bbox[conf_preds == 0])

    def __getitem__(self, index):
        file_id = self.data[index]
        label = self.labels[index]
        bbox = self.bbox[index]
        pil_bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        file_dir = os.path.join(self.img_root, file_id)
        if not file_dir.endswith('.JPG'):
            file_dir += '.jpg'

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB').crop(pil_bbox)
            sample = ImageOps.expand(sample, tuple((max(sample.size) - s) // 2 for s in list(sample.size)))

        if self.transform is not None:
            sample = self.transform(sample)

        if self.conf_preds is not None:
            conf_pred = self.conf_preds[index]
            return sample, label, conf_pred, index
        else:
            return sample, label


@register_dataset_obj('CCT_CIS_CROP_S2')
class CCT_CIS_CROP_S2(CCT_CROP_ST2):

    name = 'CCT_CIS_CROP_S2'

    def __init__(self, rootdir, class_indices, dset='train', split=None,
                 transform=None, conf_preds=None, unknown_only=False):
        super(CCT_CIS_CROP_S2, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                              split=split, transform=transform, conf_preds=conf_preds,
                                              unknown_only=unknown_only)
        ann_dir = os.path.join(self.ann_root, 'cis_{}_annotations_season_2.json'.format(dset))
        self.load_data(ann_dir)
        if unknown_only:
            self.pick_unknown()
        if split is not None:
            self.data_split()

@register_dataset_obj('CCT_TRANS_CROP')
class CCT_TRANS_CROP(CCT_CROP_ST2):

    name = 'CCT_TRANS_CROP'

    def __init__(self, rootdir, class_indices, dset='train', split=None,
                 transform=None, conf_preds=None, unknown_only=False):
        super(CCT_TRANS_CROP, self).__init__(rootdir=rootdir, class_indices=class_indices, dset=dset,
                                             split=split, transform=transform, conf_preds=conf_preds,
                                             unknown_only=unknown_only)
        if self.dset == 'train':
            ann_dir = os.path.join(self.ann_root, 'cis_{}_annotations_season_2.json'.format(dset))
        else:
            ann_dir = os.path.join(self.ann_root, 'trans_{}_annotations.json'.format(dset))

        self.load_data(ann_dir)
        if unknown_only:
            self.pick_unknown()
        if split is not None:
            self.data_split()

