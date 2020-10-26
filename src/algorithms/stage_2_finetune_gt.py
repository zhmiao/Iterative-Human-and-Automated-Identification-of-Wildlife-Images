import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm, Algorithm, stage_2_metric, acc
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model
from src.algorithms.stage_1_plain import load_data, PlainStage1


@register_algorithm('GTFineTuneStage2')
class GTFineTuneStage2(PlainStage1):

    """
    Overall training function.
    """

    name = 'GTFineTuneStage2'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(GTFineTuneStage2, self).__init__(args=args)

