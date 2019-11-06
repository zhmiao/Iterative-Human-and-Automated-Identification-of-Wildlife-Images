import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from .utils import register_algorithm
from src.data.dataloader import load_dataset
from src.models.utils import get_model


@register_algorithm('PlainResNet')
class PlainResNet:

    """
    Overall training function.
    """

    name = 'PlainResNet'

    def __init__(self, args):

        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval
        self.logger = args.logger
        self.best_acc = 0.
        self.out_file = './weights/{}/{}_{}.pth'.format(args.algorithm, args.conf_id, args.session)

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader = load_dataset(name=args.dataset_name, dset='train', rootdir=args.dataset_root,
                                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        self.testloader = load_dataset(name=args.dataset_name, dset='test', rootdir=args.dataset_root,
                                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        self.valloader = load_dataset(name=args.dataset_name, dset='test', rootdir=args.dataset_root,
                                      batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        self.net = get_model(name=args.model_name, num_cls=args.num_cls,
                             weights_init=args.weights_init, num_layers=args.num_layers)
        # print network and arguments
        print(self.net)

        ######################
        # Optimization setup #
        ######################
        # Setup optimizer parameters for each network component
        net_optim_params_list = [
            {'params': self.net.feature.parameters(),
             'lr': args.lr_feature,
             'momentum': args.momentum_feature,
             'weight_decay': args.weight_decay_feature},
            {'params': self.net.classifier.parameters(),
             'lr': args.lr_classifier,
             'momentum': args.momentum_classifier,
             'weight_decay': args.weight_decay_classifier}
        ]
        # Setup optimizer and optimizer scheduler
        self.opt_net = optim.SGD(net_optim_params_list)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt_net, step_size=args.step_size, gamma=args.gamma)

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
            if batch_idx % self.log_interval == 0:
                # compute overall acc
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean()
                # log update info
                info_str += 'Acc: {:0.1f} Xent: {:.3f}'.format(acc.item() * 100, loss.item())
                self.logger.info(info_str)

    def train(self):
        for epoch in range(self.num_epochs):

            # Training
            self.train_epoch(epoch)
            self.scheduler.step()

            # Validation
            self.logger.info('Validation.')
            eval_info, val_acc = self.evaluate(self.valloader)
            self.logger.info(eval_info)
            self.logger.info('Macro Acc: {}'.format(val_acc))
            if val_acc > self.best_acc:
                self.save()

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

        class_acc = class_correct / class_num

        eval_info = 'Per-class evaluation results: '

        for i in range(len(class_acc)):
            eval_info += '{} [{}/{}]'.format(class_acc[i], class_correct[i], class_num[i])

        return eval_info, class_acc.mean()

    def save_model(self):
        ##############
        # Save Model #
        ##############
        os.makedirs(self.out_file.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to', self.out_file)
        self.net.save(self.out_file)
