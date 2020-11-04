import os
from os import sep
import math
import numpy as np
from datetime import datetime
from tqdm import tqdm
import copy
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR               
               
from .utils import register_algorithm, LDAMLoss
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model
from src.algorithms.stage_1_plain import PlainStage1
from src.algorithms.stage_2_fixmatch_ema import load_data, EMAFixMatchStage2, ModelEMA


@register_algorithm('LDAMEMAFixMatchStage2')
class LDAMEMAFixMatchStage2(EMAFixMatchStage2):

    """
    Overall training function.
    """

    name = 'LDAMEMAFixMatchStage2'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(LDAMEMAFixMatchStage2, self).__init__(args=args)

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init, num_layers=self.args.num_layers, init_feat_only=True,
                             norm=True, parallel=True)

        self.set_optimizers()

        if self.args.ema:
            self.logger.info('\nGetting EMA model.')
            self.feature_ema = ModelEMA(self.args.lr_feature, self.args.weight_decay_feature,
                                        self.net.feature, decay=0.999)
            self.classifier_ema = ModelEMA(self.args.lr_classifier, self.args.weight_decay_classifier,
                                           self.net.classifier, decay=0.999)

    def set_optimizers(self, lr_factor=1.):
        self.logger.info('** SETTING OPTIMIZERS!!! **')
        ######################
        # Optimization setup #
        ######################
         
        # Setup optimizer parameters for each network component
        net_optim_params_list = [
            {'params': self.net.feature.parameters(),
             'lr': self.args.lr_feature * lr_factor,
             'momentum': self.args.momentum_feature,
             'weight_decay': self.args.weight_decay_feature,
             'nesterov': True},
            {'params': self.net.classifier.parameters(),
             'lr': self.args.lr_classifier * lr_factor,
             'momentum': self.args.momentum_classifier,
             'weight_decay': self.args.weight_decay_classifier,
             'nesterov': True}
        ]

        def cosine_scheduler(optimizer,
                             num_warmup_steps,
                             num_training_steps,
                             num_cycles=7./16.,
                             last_epoch=-1):
            def _lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                no_progress = float(current_step - num_warmup_steps) / \
                    float(max(1, num_training_steps - num_warmup_steps))
                return max(0., math.cos(math.pi * num_cycles * no_progress))

            return LambdaLR(optimizer, _lr_lambda, last_epoch)

        # Setup optimizer and optimizer scheduler
        self.opt_net = optim.SGD(net_optim_params_list)
        # self.scheduler = cosine_scheduler(optimizer=self.opt_net,
        #                                   num_warmup_steps=5 * self.train_iterations,
        #                                   num_training_steps=self.args.num_epochs * self.train_iterations)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt_net, step_size=self.args.step_size, gamma=self.args.gamma)

    def train(self):

        best_epoch = 0
        best_acc = 0.

        for epoch in range(self.num_epochs):

            idx = epoch // int(self.num_epochs * 2 / 3) 
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.train_annotation_counts)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.train_annotation_counts)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

            self.net.criterion_cls = LDAMLoss(cls_num_list=self.train_annotation_counts, max_m=0.3, 
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
                self.save_ema_model()

        self.logger.info('\nBest Model Appears at Epoch {}...'.format(best_epoch))
        self.save_model()

    def train_epoch(self, epoch):

        self.net.train()

        loader_l = self.trainloader_l
        loader_u = self.trainloader_u

        iter_l = iter(loader_l)
        iter_u = iter(loader_u)

        N = self.train_iterations 

        for batch_idx in range(N):

            # log basic adda train info
            info_str = '[Train FixMatch (Stage 2)] '
            info_str += 'Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                              N, 100 * batch_idx / N)

            ########################
            # Setup data variables #
            ########################

            data_l, labels_l = next(iter_l)

            batch_size = data_l.shape[0]

            try:
                (data_u_w, data_u_s), labels_u = next(iter_u)
            except StopIteration:
                iter_u = iter(loader_u)
                (data_u_w, data_u_s), labels_u = next(iter_u)

            data = torch.cat((data_l, data_u_w, data_u_s)).cuda()
            labels_l, labels_u = labels_l.cuda(), labels_u.cuda()
            data.requires_grad = False
            labels_l.requires_grad = False
            labels_u.requires_grad = False

            ####################
            # Forward and loss #
            ####################
            # forward
            feats = self.net.feature(data)
            logits = self.net.classifier(feats)
            # split logits
            logits_l = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # Pseudo labels
            pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.args.fixmatch_t).float()

            # calculate loss
            loss_l = self.net.criterion_cls(logits_l, labels_l)
            # loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            loss_u = (self.net.criterion_cls(logits_u_s, targets_u, reduction='none') * mask).mean()

            loss = loss_l + self.args.fixmatch_lambda * loss_u
            
            #############################
            # Backward and optimization #
            #############################
            # zero gradients for optimizer
            self.opt_net.zero_grad()
            # loss backpropagation
            loss.backward()
            # optimize step
            self.opt_net.step()
            self.scheduler.step()

            if self.args.ema:
                self.feature_ema.update(self.net.feature)
                self.classifier_ema.update(self.net.classifier)

            ###########
            # Logging #
            ###########
            if batch_idx % self.log_interval == 0:
                # compute overall acc
                preds_l = logits_l.argmax(dim=1)
                acc_l = (preds_l == labels_l).float().mean()
                preds_u = logits_u_w.argmax(dim=1)
                acc_u = (preds_u == labels_u).float().mean()
                # log update info
                info_str += 'Acc_l: {:0.1f} Acc_u: {:0.1f} '.format(acc_l.item() * 100, acc_u.item() * 100)
                info_str += 'Xent_l: {:.3f} Xent_u: {:.3f}'.format(loss_l.item(), loss_u.item())
                self.logger.info(info_str)

