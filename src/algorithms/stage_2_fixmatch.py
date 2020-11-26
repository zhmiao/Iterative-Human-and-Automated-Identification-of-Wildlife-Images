import os
from os import sep
import math
import numpy as np
from datetime import datetime
from tqdm import tqdm
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR               
               
from .utils import register_algorithm
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model
from src.algorithms.stage_1_plain import PlainStage1


def load_data(args, cas, conf_preds):

    print('Using class indices: {} \n'.format(class_indices[args.class_indices]))

    cls_idx = class_indices[args.class_indices]

    trainloader_l = load_dataset(name=args.dataset_name,
                                 class_indices=cls_idx,
                                 dset='train',
                                 transform=args.train_transform,
                                 rootdir=args.dataset_root,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 cas_sampler=cas,
                                 conf_preds=conf_preds,
                                 GTPS_mode='GT')

    trainloader_u = load_dataset(name=args.dataset_name,
                                 class_indices=cls_idx,
                                 dset='train',
                                 transform=args.train_transform + '_Unlabeled',
                                 rootdir=args.dataset_root,
                                 batch_size=args.batch_size * 4,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 cas_sampler=cas,
                                 conf_preds=conf_preds,
                                 GTPS_mode='PS')

    valloader = load_dataset(name=args.dataset_name,
                             class_indices=cls_idx,
                             dset='val',
                             transform='eval',
                             rootdir=args.dataset_root,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             cas_sampler=False)

    valloaderunknown = load_dataset(name=args.unknown_dataset_name,
                                    class_indices=cls_idx,
                                    dset='val',
                                    transform='eval',
                                    rootdir=args.dataset_root,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    cas_sampler=False)

    deployloader = load_dataset(name=args.deploy_dataset_name,
                                class_indices=cls_idx,
                                dset=None,
                                transform='eval',
                                rootdir=args.dataset_root,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                cas_sampler=False)

    return trainloader_l, trainloader_u, valloader, valloaderunknown, deployloader


@register_algorithm('FixMatchStage2')
class FixMatchStage2(PlainStage1):

    """
    Overall training function.
    """

    name = 'FixMatchStage2'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        self.args = args
        self.logger = self.args.logger
        self.weights_path = './weights/{}/{}_{}.pth'.format(self.args.algorithm, self.args.conf_id, self.args.session)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        #######################################
        # Setup data for training and testing #
        #######################################
        # self.conf_preds = list(np.fromfile(args.weights_init.replace('_ft.pth', '_conf_preds.npy')).astype(int))
        self.conf_preds = list(np.fromfile(args.weights_init.replace('.pth', '_conf_preds.npy')).astype(int))

        (self.trainloader_l, self.trainloader_u,
         self.valloader, self.valloaderunknown, self.deployloader) = load_data(args, cas=False,
                                                                               conf_preds=self.conf_preds)

        self.train_class_counts = self.trainloader_l.dataset.class_counts
        self.train_annotation_counts = self.trainloader_l.dataset.class_counts_ann

        self.train_iterations = len(self.trainloader_l)

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init, num_layers=self.args.num_layers, init_feat_only=False,
                             parallel=True)

        self.set_optimizers()

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
        self.scheduler = cosine_scheduler(optimizer=self.opt_net,
                                          num_warmup_steps=5 * self.train_iterations,
                                          num_training_steps=self.args.num_epochs * self.train_iterations)

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
            loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

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
