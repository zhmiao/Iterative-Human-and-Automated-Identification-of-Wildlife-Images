import os
import json

import torch
from torch.utils.data import Dataset

from .dataloader import register_dataset_obj


@register_dataset_obj('CCT_15')
class CCT_15(Dataset):

    def __init__(self, rootdir, dset='train', transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
