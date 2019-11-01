import os
import json

import torch
from torch.utils.data import Dataset

from .dataloader import register_dataset_obj


@register_dataset_obj('CCT_trans')
class CCT_trans(Dataset):

    name = 'CCT_trans'

    def __init__(self, rootdir, dset='train', transform=None):
        self.root = rootdir

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


@register_dataset_obj('CCT_cis')
class CCT_cis(Dataset):

    name = 'CCT_cis'

    def __init__(self, rootdir, dset='train', transform=None):
        self.root = rootdir

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


