import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from .utils import register_dataset_obj

class MOZ(Dataset):

    def __init__(self, rootdir, dset='train', transform=None):
        self.data_root = os.path.join(rootdir, 'Mozambique')
        self.dset = dset
        self.transform = transform
        self.data = None
        self.categories_names = {}
        self.categories_labels = {}

    def class_counts_cal(self):
        labels = []
        pass

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

    def __init__(self, rootdir, dset='train', transform=None):
        super(MOZ_S1, self).__init__(rootdir=rootdir, dset=dset, transform=transform)


@register_dataset_obj('MOZ_S2')
class MOZ_S2(MOZ):

    name = 'MOZ_S2'

    def __init__(self, rootdir, dset='train', transform=None):
        super(MOZ_S2, self).__init__(rootdir=rootdir, dset=dset, transform=transform)

