import os
from os import sep
import numpy as np
from datetime import datetime
from tqdm import tqdm
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from .utils import register_algorithm, LDAMLoss
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model
from src.algorithms.stage_2_pslabel import SemiStage2


@register_algorithm('LDAMSemiStage2_TUNE2')
class LDAMSemiStage2_TUNE2(SemiStage2):

    """
    Overall training function.
    """

    name = 'LDAMSemiStage2_TUNE2'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(LDAMSemiStage2_TUNE2, self).__init__(args=args)

    def set_train(self):
        # setup network
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init,
                             num_layers=self.args.num_layers, init_feat_only=False,
                             norm=True)

        self.set_optimizers(lr_factor=1.)

        self.logger.info('\nMaking all pseudolabels NONE..')
        self.pseudo_labels_soft = None
        self.pseudo_labels_hard = None

        self.logger.info('\nUpdating Current Pseudo Labels..')
        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.pseudo_label_reset(self.trainloader_eval, soft_reset=False, hard_reset=True)

        self.reset_trainloader(pseudo_hard=self.pseudo_labels_hard,
                               pseudo_soft=self.pseudo_labels_soft)

    def train(self):

        best_epoch = 0
        best_acc = 0.
        best_semi_iter = 0

        for semi_i in range(self.args.semi_iters):

            for epoch in range(self.args.num_epochs):
                
                idx = 0 
                betas = [0.9999]
                effective_num = 1.0 - np.power(betas[idx], self.train_annotation_counts)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.train_annotation_counts)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

                self.net.criterion_cls_hard = LDAMLoss(cls_num_list=self.train_annotation_counts, max_m=0.3, 
                                                       s=30, weight=per_cls_weights).cuda()

                self.train_epoch(epoch, soft=(self.pseudo_labels_soft is not None))

                # Validation
                self.logger.info('\nValidation.')
                val_acc_mac = self.evaluate(self.valloader, ood=False)
                if val_acc_mac > best_acc:
                    self.logger.info('\nUpdating Best Model Weights!!')
                    self.net.update_best()
                    best_acc = val_acc_mac
                    best_epoch = epoch
                    best_semi_iter = semi_i

                self.logger.info('\nCurrrent Best Acc is {:.3f} at epoch {} semi-iter {}...'
                                 .format(best_acc * 100, best_epoch, best_semi_iter))

            self.save_model()

            # Revert to best weights
            self.net.load_state_dict(copy.deepcopy(self.net.best_weights))

            # Get original pseudo labels
            self.pseudo_labels_hard = np.fromfile('./weights/EnergyStage1/101920_MOZ_S1_0_init_pseudo_hard.npy',
                                                  dtype=np.int)

            # Reseting pseudo labels with best model
            self.pseudo_label_reset(self.trainloader_eval,
                                    soft_reset=(self.pseudo_labels_soft is not None), 
                                    hard_reset=True)

            # Reseting train loaders with new pseudolabels 
            self.reset_trainloader(pseudo_hard=self.pseudo_labels_hard,
                                   pseudo_soft=self.pseudo_labels_soft)

            # Reseting optimizers
            self.set_optimizers(lr_factor=1.)

            self.logger.info('\nBest Model Appears at Epoch {} Semi-iteration {} with Acc {:.3f}...'
                             .format(best_epoch, best_semi_iter, best_acc * 100))