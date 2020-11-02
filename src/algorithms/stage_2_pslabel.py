import os
from os import sep
import numpy as np
from datetime import datetime
from tqdm import tqdm
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model
from src.algorithms.stage_1_plain import PlainStage1


def load_val_data(args):

    print('Using class indices: {} \n'.format(class_indices[args.class_indices]))

    cls_idx = class_indices[args.class_indices]

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

    return valloader, valloaderunknown, deployloader


def load_train_data(args, conf_preds, pseudo_labels_hard, pseudo_labels_soft, cas=False):

    print('Using class indices: {} \n'.format(class_indices[args.class_indices]))

    cls_idx = class_indices[args.class_indices]

    trainloader_gtps = load_dataset(name=args.dataset_name,
                                    class_indices=cls_idx,
                                    dset='train',
                                    transform='eval',
                                    rootdir=args.dataset_root,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    cas_sampler=False,
                                    conf_preds=conf_preds,
                                    pseudo_labels_hard=pseudo_labels_hard,
                                    pseudo_labels_soft=None,
                                    GTPS_mode='both')

    trainloader_gt = load_dataset(name=args.dataset_name,
                                  class_indices=cls_idx,
                                  dset='train',
                                  transform=args.train_transform,
                                  rootdir=args.dataset_root,
                                  batch_size=int(args.batch_size / 2),
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  cas_sampler=cas,
                                  conf_preds=conf_preds,
                                  pseudo_labels_hard=pseudo_labels_hard,
                                  pseudo_labels_soft=pseudo_labels_soft,
                                  GTPS_mode='GT')

    trainloader_ps = load_dataset(name=args.dataset_name,
                                  class_indices=cls_idx,
                                  dset='train',
                                  transform=args.train_transform,
                                  rootdir=args.dataset_root,
                                  batch_size=int(args.batch_size / 2),
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  cas_sampler=cas,
                                  conf_preds=conf_preds,
                                  pseudo_labels_hard=pseudo_labels_hard,
                                  pseudo_labels_soft=pseudo_labels_soft,
                                  GTPS_mode='PS')

    return trainloader_gtps, trainloader_gt, trainloader_ps


@register_algorithm('SemiStage2')
class SemiStage2(PlainStage1):

    """
    Overall training function.
    """

    name = 'SemiStage2'
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
        self.valloader, self.valloaderunknown, self.deployloader = load_val_data(args)

        self.conf_preds = list(np.fromfile(args.weights_init.replace('_ft.pth', '_conf_preds.npy')).astype(int))
        self.pseudo_labels_hard = np.fromfile(args.weights_init.replace('_ft.pth', '_init_pseudo_hard.npy'),
                                              dtype=np.int)
        self.pseudo_labels_soft = None

        self.reset_trainloader(pseudo_hard=self.pseudo_labels_hard,
                               pseudo_soft=self.pseudo_labels_soft)

        self.train_class_counts = self.trainloader_gtps.dataset.class_counts
        self.train_annotation_counts = self.trainloader_gtps.dataset.class_counts_ann

    def reset_trainloader(self, pseudo_hard, pseudo_soft, cas=False):
        self.logger.info('\nReseting training loader and sampler with pseudo labels.')
        self.logger.info('\nTRAINLOADER_NO_UP_GT....')
        (self.trainloader_gtps,
         self.trainloader_gt,
         self.trainloader_ps) = load_train_data(self.args, self.conf_preds, 
                                                pseudo_hard, pseudo_soft, cas=cas)

    def set_train(self):
        # setup network
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init,
                             num_layers=self.args.num_layers, init_feat_only=True,
                             T=self.args.T, alpha=self.args.alpha)

        self.set_optimizers(lr_factor=1.)

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
             'weight_decay': self.args.weight_decay_feature},
            {'params': self.net.classifier.parameters(),
             'lr': self.args.lr_classifier * lr_factor,
             'momentum': self.args.momentum_classifier,
             'weight_decay': self.args.weight_decay_classifier}
        ]
        # Setup optimizer and optimizer scheduler
        self.opt_net = optim.SGD(net_optim_params_list)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt_net, step_size=self.args.step_size, gamma=self.args.gamma)

    def pseudo_label_reset(self, loader, soft_reset=False, hard_reset=False):
        self.net.eval()
        total_preds, total_labels, total_logits, conf_preds = self.evaluate_forward(loader, ood=False, out_conf=True)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        total_logits = np.concatenate(total_logits, axis=0)
        if soft_reset:
            self.logger.info("** Reseting soft pseudo labels **\n")
            if self.pseudo_labels_soft is not None:
                self.pseudo_labels_soft[conf_preds == 1] = total_logits[conf_preds == 1]
            else:
                self.pseudo_labels_soft = total_logits
        if hard_reset:
            self.logger.info("** Reseting hard pseudo labels **\n")
            self.pseudo_labels_hard[conf_preds == 1] = total_preds[conf_preds == 1]
            pseudo_hard_path = self.weights_path.replace('.pth', '_pseudo_hard.npy')
            self.logger.info('Saving updated hard pseudo labels to {}'.format(pseudo_hard_path))
            self.pseudo_labels_hard.tofile(pseudo_hard_path)

    def train(self):

        best_epoch = 0
        best_acc = 0.

        for epoch in range(self.args.num_epochs):

            self.train_epoch(epoch, soft=(self.pseudo_labels_soft is not None))

            # Validation
            self.logger.info('\nValidation.')
            val_acc_mac = self.evaluate(self.valloader, ood=False)
            if val_acc_mac > best_acc:
                self.logger.info('\nUpdating Best Model Weights!!')
                self.net.update_best()
                best_acc = val_acc_mac
                best_epoch = epoch

                # Reset pseudo labels
                self.logger.info('\nUpdating Current Pseudo Labels..')
                self.pseudo_label_reset(self.trainloader_gtps,
                                        soft_reset=(self.pseudo_labels_soft is not None), 
                                        hard_reset=True)

            self.logger.info('\nCurrrent Best Acc is {:.3f} at epoch {}...'
                             .format(best_acc * 100, best_epoch))

        self.logger.info('\nBest Model Appears at Epoch {} with Acc {:.3f}...'
                         .format(best_epoch, best_acc * 100))

        self.save_model()
        self.pseudo_label_reset(self.trainloader_gtps,
                                soft_reset=(self.pseudo_labels_soft is not None), 
                                hard_reset=True)

    def train_epoch(self, epoch, soft=False):

        self.net.train()

        loader_gt = self.trainloader_gt
        loader_ps = self.trainloader_ps

        iter_gt = iter(loader_gt)
        iter_ps = iter(loader_ps)

        N = max(len(self.trainloader_gtps) * 1.5, len(iter_ps))
        

        for batch_idx in range(N):

            # log basic adda train info
            info_str = '[Train Semi (Stage 2)] '
            info_str += '[Soft] ' if soft else '[Hard] '
            info_str += 'Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                              N, 100 * batch_idx / N)

            ########################
            # Setup data variables #
            ########################
            try:
                input_gt = next(iter_gt)
            except StopIteration:
                iter_gt = iter(loader_gt)
                input_gt = next(iter_gt)

            input_ps = next(iter_ps)

            if soft:
                data_gt, labels_gt, soft_target_gt = input_gt 
                data_ps, labels_ps, soft_target_ps = input_ps
                soft_target = torch.cat((soft_target_gt, soft_target_ps), dim=0).cuda()
                soft_target.requires_grad = False
            else:
                data_gt, labels_gt = input_gt 
                data_ps, labels_ps = input_ps
                soft_target = None

            data = torch.cat((data_gt, data_ps), dim=0).cuda()
            labels = torch.cat((labels_gt, labels_ps), dim=0).cuda()
            data.requires_grad = False
            labels.requires_grad = False

            ####################
            # Forward and loss #
            ####################
            # forward
            feats = self.net.feature(data)
            logits = self.net.classifier(feats)
            # calculate loss
            if soft:
                loss = self.net.criterion_cls_soft(logits, labels, soft_target)
            else:
                loss = self.net.criterion_cls_hard(logits, labels)

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

    def evaluate_forward(self, loader, ood=False, out_conf=False):
        total_preds = []
        total_labels = []
        total_logits = []
        total_energy_score = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for data, labels in tqdm(loader, total=len(loader)):

                # setup data
                data, labels = data.cuda(), labels.cuda()
                data.requires_grad = False
                labels.requires_grad = False

                # forward
                feats = self.net.feature(data)
                logits = self.net.classifier(feats)

                _, preds = F.softmax(logits, dim=1).max(dim=1)

                energy_score = -(self.args.energy_T * torch.logsumexp(logits / self.args.energy_T, dim=1))

                if ood:
                    # Set unconfident prediction to -1
                    preds[-energy_score <= self.args.energy_the] = -1

                total_preds.append(preds.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())
                total_logits.append(logits.detach().cpu().numpy())
                total_energy_score.append(energy_score.detach().cpu().numpy())

        if out_conf:
            total_energy_score = np.concatenate(total_energy_score, axis=0)
            conf_preds = np.zeros(len(total_energy_score))
            conf_preds[-total_energy_score > self.args.energy_the] = 1
            return total_preds, total_labels, total_logits, conf_preds
        else:
            return total_preds, total_labels, total_logits