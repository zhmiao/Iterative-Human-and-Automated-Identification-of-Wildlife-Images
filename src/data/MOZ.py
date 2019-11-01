import os

import torch
from torch.utils.data import Dataset

from .dataloader import register_dataset_obj


@register_dataset_obj('MOZ_S1')
class MOZ_S1(Dataset):

    name = 'MOZ_S1'

    def __init__(self, rootdir, dset='train', transform=None):
        self.root = rootdir

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


@register_dataset_obj('MOZ_S2')
class MOZ_S2(Dataset):

    name = 'MOZ_S2'

    def __init__(self, rootdir, dset='train', transform=None):
        self.root = rootdir

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
