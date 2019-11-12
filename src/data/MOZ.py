import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from .utils import register_dataset_obj

# Label indicies for two seasons
label_indicies = {
    'Aardvark': 0,
    'Baboon': 1,
    'Buffalo': 2,
    'Bushbuck': 3,
    'Bushpig': 4,
    'Civet': 5,
    'Duiker_common': 6,  # Only in season 1
    'Duiker_red': 7,
    'Elephant': 8,
    'Genet': 9,
    'Guineafowl_helmeted': 10,
    'Hare': 11,
    'Hartebeest': 12,
    'Honey_badger': 13,
    'Hornbill_ground': 14,
    'Impala': 15,
    'Kudu': 16,
    'Mongoose_banded': 17,
    'Mongoose_bushy_tailed': 18,
    'Mongoose_large_grey': 19,
    'Mongoose_marsh': 20,
    'Mongoose_slender': 21,
    'Mongoose_white_tailed': 22,
    'Nyala': 23,
    'Oribi': 24,
    'Porcupine': 25,
    'Reedbuck': 26,
    'Sable_antelope': 27,
    'Samango': 28,  # Only in season 1
    'Vervet': 29,
    'Warthog': 30,
    'Waterbuck': 31,
    'Wildebeest': 32,
    'Bushbaby': 33  # Only in season 2
}


class MOZ(Dataset):

    def __init__(self, rootdir, dset='train', transform=None):
        self.data_root = os.path.join(rootdir, 'Mozambique')
        self.dset = dset if dset != 'val' else 'test'  # No separate validation set for now
        self.transform = transform
        self.season = None
        self.data = []
        self.labels = []

    def load_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                line_sp = line.replace('\n', '').split(' ')
                assert self.season is not None
                assert line_sp[1] in label_indicies.keys()
                data_path = os.path.join(self.root, 'Mozambique_season_{}'.format(self.season), line_sp[0])
                self.data.append(data_path)
                self.labels.append(label_indicies[line_sp[1]])

    def class_counts_cal(self):
        labels = []
        label_counts = np.array([0 for _ in range(len(np.unique(self.labels))])
        for entry in self.data:
            labels.append(self.categories_labels[entry['category_id']])

        unique_labels, unique_counts = np.unique(labels, return_counts=True)
        for i in range(len(unique_labels)):
            label_counts[unique_labels[i]] = unique_counts[i]
        return unique_labels, label_counts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = None
        file_dir = None

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


@register_dataset_obj('MOZ_S1')
class MOZ_S1(MOZ):

    name = 'MOZ_S1'
    class_num = 33

    def __init__(self, rootdir, dset='train', transform=None):
        super(MOZ_S1, self).__init__(rootdir=rootdir, dset=dset, transform=transform)
        self.season = 1
        self.load_txt(os.path.join(self.root, 'SplitLists', '{}_season_{}.txt'.format(self.dset, self.season)))


@register_dataset_obj('MOZ_S2')
class MOZ_S2(MOZ):

    name = 'MOZ_S2'
    class_num = 32  # 34 classes in totall including season 1 classes

    def __init__(self, rootdir, dset='train', transform=None):
        super(MOZ_S2, self).__init__(rootdir=rootdir, dset=dset, transform=transform)
        self.season = 2
        self.load_txt(os.path.join(self.root, 'SplitLists', '{}_season_{}.txt'.format(self.dset, self.season)))

