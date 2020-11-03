import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model
from src.algorithms.stage_1_plain import PlainStage1
from src.algorithms.stage_2_finetune_gt import load_dataset, GTFineTuneStage2
from src.algorithms.utils import LDAMLoss, register_algorithm



@register_algorithm('LDAMGTFineTuneStage2')
class LDAMGTFineTuneStage2(GTFineTuneStage2):

    """
    Overall training function.
    """

    name = 'LDAMGTFineTuneStage2'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(LDAMGTFineTuneStage2, self).__init__(args=args)

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init, num_layers=self.args.num_layers, init_feat_only=True,
                             norm=True)

        self.set_optimizers()

    def train(self):

        best_epoch = 0
        best_acc = 0.

        for epoch in range(self.num_epochs):
            
            idx = epoch // int(self.num_epochs / 2) 
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.train_annotation_counts)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.train_annotation_counts)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

            self.net.criterion_cls = LDAMLoss(cls_num_list=self.train_annotation_counts, max_m=0.5, 
                                              s=30, weight=per_cls_weights).cuda()

            # Training
            self.train_epoch(epoch)

            # Validation
            self.logger.info('\nValidation.')
            val_acc_mac = self.evaluate(self.valloader, ood=False)
            if val_acc_mac > best_acc:
                self.logger.info('\nUpdating Best Model Weights!!')
                self.net.update_best()
                best_acc = val_acc_mac
                best_epoch = epoch

        self.logger.info('\nBest Model Appears at Epoch {}...'.format(best_epoch))
        self.save_model()

