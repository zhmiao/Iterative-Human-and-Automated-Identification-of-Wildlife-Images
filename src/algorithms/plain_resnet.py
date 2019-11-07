import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim

from .utils import register_algorithm, Algorithm
from src.data.utils import load_dataset
from src.models.utils import get_model


@register_algorithm('PlainResNet')
class PlainResNet(Algorithm):

    """
    Overall training function.
    """

    name = 'PlainResNet'

    def __init__(self, args):
        super(PlainResNet, self).__init__(args=args)

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader = load_dataset(name=self.args.dataset_name, dset='train', rootdir=self.args.dataset_root,
                                        batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

        self.testloader = load_dataset(name=self.args.dataset_name, dset='test', rootdir=self.args.dataset_root,
                                       batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        self.valloader = load_dataset(name=self.args.dataset_name, dset='val', rootdir=self.args.dataset_root,
                                      batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        _, self.train_num = self.trainloader.dataset.class_num_cal()

        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        self.net = get_model(name=self.args.model_name, num_cls=self.args.num_cls,
                             weights_init=self.args.weights_init, num_layers=self.args.num_layers)
        # print network and arguments
        print(self.net)

        ######################
        # Optimization setup #
        ######################
        # Setup optimizer parameters for each network component
        net_optim_params_list = [
            {'params': self.net.feature.parameters(),
             'lr': self.args.lr_feature,
             'momentum': self.args.momentum_feature,
             'weight_decay': self.args.weight_decay_feature},
            {'params': self.net.classifier.parameters(),
             'lr': self.args.lr_classifier,
             'momentum': self.args.momentum_classifier,
             'weight_decay': self.args.weight_decay_classifier}
        ]
        # Setup optimizer and optimizer scheduler
        self.opt_net = optim.SGD(net_optim_params_list)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt_net, step_size=self.args.step_size, gamma=self.args.gamma)

    def train_epoch(self, epoch):

        self.net.train()

        N = len(self.trainloader)

        for batch_idx, (data, labels) in enumerate(self.trainloader):

            # log basic adda train info
            info_str = '[Train plain] Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                                           N, 100 * batch_idx / N)

            ########################
            # Setup data variables #
            ########################
            data, labels = data.cuda(), labels.cuda()

            data.require_grad = False
            labels.require_grad = False

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
            if batch_idx % self.args.log_interval == 0:
                # compute overall acc
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean()
                # log update info
                info_str += 'Acc: {:0.1f} Xent: {:.3f}'.format(acc.item() * 100, loss.item())
                self.logger.info(info_str)

    def train(self):

        best_acc = 0.

        for epoch in range(self.args.num_epochs):

            # Training
            self.train_epoch(epoch)
            self.scheduler.step()

            # Validation
            self.logger.info('Validation.')
            eval_info, val_acc = self.evaluate(self.valloader)
            self.logger.info(eval_info)
            self.logger.info('Macro Acc: {}'.format(val_acc))
            if val_acc > best_acc:
                self.save_model()

    def evaluate(self, loader):

        self.net.eval()

        classes, class_num = loader.dataset.class_num_cal()
        class_correct = np.array([0. for _ in range(len(classes))])

        with torch.set_grad_enabled(False):

            for data, labels in tqdm(loader, total=len(loader)):

                # setup data
                data, labels = data.cuda(), labels.cuda()
                data.require_grad = False
                labels.require_grad = False

                # forward
                feats = self.net.feature(data)
                logits = self.net.classifier(feats)

                # compute correct
                preds = logits.argmax(dim=1)
                for i in range(len(preds)):
                    pred = preds[i]
                    label = labels[i]
                    if pred == label:
                        class_correct[label] += 1

        # Record accuracies
        class_acc = class_correct / class_num
        eval_info = '{} Per-class evaluation results: '.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        for i in range(len(class_acc)):
            eval_info += '{} ({}): {}'.format(classes[i], self.train_num[i], class_acc[i])

        return eval_info, class_acc.mean()

    def save_model(self):
        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path))
        self.net.save(self.weights_path)
