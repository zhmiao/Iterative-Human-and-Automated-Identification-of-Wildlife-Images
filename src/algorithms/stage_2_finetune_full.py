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


@register_algorithm('FullFineTuneStage2')
class FullFineTuneStage2(PlainStage1):

    """
    Overall training function.
    """

    name = 'FullFineTuneStage2'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(FullFineTuneStage2, self).__init__(args=args)

    def train_epoch(self, epoch):

        self.net.train()

        N = len(self.trainloader)

        for batch_idx, (data, labels) in enumerate(self.trainloader):

            # log basic adda train info
            info_str = '[FineTuning FULL {} - Stage 2] Epoch: {} [{}/{} ({:.2f}%)] '.format(self.net.name, epoch, batch_idx,
                                                                                            N, 100 * batch_idx / N)

            ########################
            # Setup data variables #
            ########################
            data, labels = data.cuda(), labels.cuda()
            data.requires_grad = False
            labels.requires_grad = False

            ####################
            # Forward and loss #
            ####################
            # forward
            feats = self.net.feature(data)
            logits = self.net.classifier(feats)
            # calculate loss
            loss = self.net.criterion_cls(logits, labels)

            #############################
            # Backward and optimization #
            #############################
            # zero gradients for optimizer
            self.opt_net.zero_grad()
            # loss backpropagation
            loss.backward()
            # optimize step
            self.opt_net.step()

            ###########
            # Logging #
            ###########
            if batch_idx % self.log_interval == 0:
                # compute overall acc
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean()
                # log update info
                info_str += 'Acc: {:0.1f} Xent: {:.3f}'.format(acc.item() * 100, loss.item())
                self.logger.info(info_str)

        self.scheduler.step()
