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
from src.algorithms.stage_2_pslabel import load_train_data, load_val_data, SemiStage2


@register_algorithm('LDAMSemiStage2_TUNE3')
class LDAMSemiStage2_TUNE3(SemiStage2):

    """
    Overall training function.
    """

    name = 'LDAMSemiStage2_TUNE3'
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
        self.conf_preds = list(np.fromfile('./weights/EnergyStage1/101920_MOZ_S1_0_conf_preds.npy').astype(int))
        (self.trainloader_eval, self.valloader, 
         self.valloaderunknown, self.deployloader) = load_val_data(args, self.conf_preds)

        self.pseudo_labels_hard_head = np.fromfile('./weights/EnergyStage1/101920_MOZ_S1_0_init_pseudo_hard.npy',
                                                   dtype=np.int)
        self.pseudo_labels_hard_tail = None

        self.pseudo_labels_soft = None

        self.train_class_counts = self.trainloader_eval.dataset.class_counts
        self.train_annotation_counts = self.trainloader_eval.dataset.class_counts_ann

    def set_train(self):
        # setup network
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init,
                             num_layers=self.args.num_layers, init_feat_only=False,
                             norm=True)

        self.set_optimizers(lr_factor=1.)

    def pseudo_label_reset(self, loader):
        self.net.eval()
        total_preds, total_labels, total_logits, conf_preds = self.evaluate_forward(loader, ood=False, out_conf=True)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        total_logits = np.concatenate(total_logits, axis=0)

        self.logger.info("** Reseting head pseudo labels **\n")
        self.pseudo_labels_hard_head[conf_preds == 1] = total_preds[conf_preds == 1]
        
        self.logger.info("** Reseting head pseudo labels **\n")
        if self.pseudo_labels_hard_tail is not None:
            self.pseudo_labels_hard_tail[conf_preds == 1] = total_preds[conf_preds == 1]
        else:
            self.pseudo_labels_hard_tail = total_preds

        # pseudo_hard_path = self.weights_path.replace('.pth', '_pseudo_hard.npy')
        # self.logger.info('Saving updated hard pseudo labels to {}'.format(pseudo_hard_path))
        # self.pseudo_labels_hard.tofile(pseudo_hard_path)

    def train(self):

        best_epoch = 0
        best_acc_mac = 0.
        best_acc_mic = 0.
        best_semi_iter = 0

        for semi_i in range(self.args.semi_iters):

            self.logger.info('\nUpdating Current Pseudo Labels..')
            os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
            self.pseudo_label_reset(self.trainloader_eval)

            for epoch in range(self.args.num_epochs):
                
                betas = [0, 0.9999]

                if epoch % 3 == 0:
                    idx = 1 
                    self.logger.info('\nUSING DRW..')
                    self.reset_trainloader(pseudo_hard=self.pseudo_labels_hard_tail,
                                           pseudo_soft=None)
                else:
                    idx = 0
                    self.logger.info('\nNO DRW..')
                    self.reset_trainloader(pseudo_hard=self.pseudo_labels_hard_head,
                                           pseudo_soft=None)

                effective_num = 1.0 - np.power(betas[idx], self.train_annotation_counts)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.train_annotation_counts)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

                self.net.criterion_cls_hard = LDAMLoss(cls_num_list=self.train_annotation_counts, max_m=0.7, 
                                                       s=30, weight=per_cls_weights).cuda()


                self.train_epoch(epoch, soft=(self.pseudo_labels_soft is not None))

                # Validation
                self.logger.info('\nValidation.')
                val_acc_mac, val_acc_mic = self.evaluate(self.valloader, ood=False)
                if ((val_acc_mac > best_acc_mac and val_acc_mic > 0.82 and semi_i > 0) 
                    or (val_acc_mac > best_acc_mac and semi_i == 0)):
                    self.logger.info('\nUpdating Best Model Weights!!')
                    self.net.update_best()
                    best_acc_mac = val_acc_mac
                    best_acc_mic = val_acc_mic
                    best_epoch = epoch
                    best_semi_iter = semi_i

                self.logger.info('\nCurrrent Best Mac Acc is {:.3f} (mic {:.3f}) at epoch {} semi-iter {}...'
                                 .format(best_acc_mac * 100, best_acc_mic * 100, best_epoch, best_semi_iter))

            self.save_model()

            # Revert to best weights
            self.net.load_state_dict(copy.deepcopy(self.net.best_weights))

            # Get original pseudo labels
            self.pseudo_labels_hard = np.fromfile('./weights/EnergyStage1/101920_MOZ_S1_0_init_pseudo_hard.npy',
                                                  dtype=np.int)

            # Reseting pseudo labels with best model
            self.pseudo_label_reset(self.trainloader_eval)

            # Reseting train loaders with new pseudolabels 
            self.reset_trainloader(pseudo_hard=self.pseudo_labels_hard,
                                   pseudo_soft=self.pseudo_labels_soft)

            # Reseting optimizers
            self.set_optimizers(lr_factor=1.)

            self.logger.info('\nCurrrent Best Mac Acc is {:.3f} (mic {:.3f}) at epoch {} semi-iter {}...'
                             .format(best_acc_mac * 100, best_acc_mic * 100, best_epoch, best_semi_iter))