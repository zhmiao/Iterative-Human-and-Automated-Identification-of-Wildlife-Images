import os
import numpy as np
from datetime import datetime
from torch import log
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
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

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.logger.info('\nLoading from {}'.format(self.weights_path))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.weights_path, num_layers=self.args.num_layers, init_feat_only=False,
                             norm=True)

    def train(self):

        best_epoch = 0
        best_acc = 0.

        for epoch in range(self.num_epochs):

            # if epoch < 3:
            #     scale = 30.
            #     self.net.criterion_cls = nn.CrossEntropyLoss()
            # elif epoch % 3 == 0:
            #     scale = 1.
            #     idx = 0 
            #     betas = [0, 0.9999]
            #     effective_num = 1.0 - np.power(betas[idx], self.train_annotation_counts)
            #     per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            #     per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.train_annotation_counts)
            #     per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

            #     self.net.criterion_cls = LDAMLoss(cls_num_list=self.train_annotation_counts, max_m=0.7, 
            #                                       s=30, weight=per_cls_weights).cuda()
            # else:
            #     scale = 1.
            #     idx = epoch // int(self.num_epochs * 2 / 3) 
            #     # idx = epoch // int(self.num_epochs / 3) 
            #     betas = [0, 0.9999]
            #     effective_num = 1.0 - np.power(betas[idx], self.train_annotation_counts)
            #     per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            #     per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.train_annotation_counts)
            #     per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

            #     self.net.criterion_cls = LDAMLoss(cls_num_list=self.train_annotation_counts, max_m=0.7, 
            #                                       s=30, weight=per_cls_weights).cuda()
            
            if epoch < 3 or epoch % 3 == 0:
                scale = 1.
                self.net.criterion_cls = nn.CrossEntropyLoss()
            else:
                scale = 1.
                idx = epoch // int(self.num_epochs / 2) 
                # idx = epoch // int(self.num_epochs / 3) 
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], self.train_annotation_counts)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.train_annotation_counts)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

                self.net.criterion_cls = LDAMLoss(cls_num_list=self.train_annotation_counts, max_m=0.3, 
                                                  s=30, weight=per_cls_weights).cuda()
            # Training
            self.train_epoch(epoch, logit_scale=scale)

            # Validation
            self.logger.info('\nValidation.')
            val_acc_mac, val_acc_mic = self.evaluate(self.valloader, ood=False)
            if val_acc_mac > best_acc:
                self.logger.info('\nUpdating Best Model Weights!!')
                self.net.update_best()
                best_acc = val_acc_mac
                best_epoch = epoch

        self.logger.info('\nBest Model Appears at Epoch {}...'.format(best_epoch))
        self.save_model()

    def train_epoch(self, epoch, logit_scale=1):

        self.net.train()

        N = len(self.trainloader)

        for batch_idx, (data, labels) in enumerate(self.trainloader):

            # log basic adda train info
            info_str = '[LDAM FT GT {} - Stage 2] Epoch: {} [{}/{} ({:.2f}%)] '.format(self.net.name, epoch, batch_idx,
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
            loss = self.net.criterion_cls(logits * logit_scale, labels)

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