import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim

from .utils import register_algorithm, Algorithm
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model


def load_data(args):

    """
    Dataloading function. This function can change alg by alg as well.
    """

    print('Using class indices: {} \n'.format(class_indices[args.class_indices]))

    trainloader = load_dataset(name=args.dataset_name,
                               class_indices=class_indices[args.class_indices],
                               dset='train',
                               rootdir=args.dataset_root,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers)

    testloader = load_dataset(name=args.dataset_name,
                              class_indices=class_indices[args.class_indices],
                              dset='test',
                              rootdir=args.dataset_root,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)

    valloader = load_dataset(name=args.dataset_name,
                             class_indices=class_indices[args.class_indices],
                             dset='val',
                             rootdir=args.dataset_root,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    return trainloader, testloader, valloader


@register_algorithm('PlainResNet')
class PlainResNet(Algorithm):

    """
    Overall training function.
    """

    name = 'PlainResNet'

    def __init__(self, args):
        super(PlainResNet, self).__init__(args=args)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader, self.testloader, self.valloader = load_data(args)
        _, self.train_class_counts = self.trainloader.dataset.class_counts_cal()

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

        best_acc = 0.

        for epoch in range(self.num_epochs):

            # Training
            self.train_epoch(epoch)
            self.scheduler.step()

            # Validation
            self.logger.info('\nValidation.')
            eval_info, val_acc_mac, val_acc_mic = self.evaluate(self.valloader)
            self.logger.info(eval_info)
            self.logger.info('Macro Acc: {:.3f}; Micro Acc: {:.3f}\n'.format(val_acc_mac, val_acc_mic))
            if val_acc_mac > best_acc:
                self.net.update_best()

        self.save_model()

    def evaluate(self, loader):

        self.net.eval()

        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()
        class_correct = np.array([0. for _ in range(len(eval_class_counts))])

        # Forward and record # correct predictions of each class
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

        # Record per class accuracies
        class_acc = class_correct[loader_uni_class] / eval_class_counts[loader_uni_class]
        overall_acc = class_correct.sum() / eval_class_counts.sum()
        eval_info = '{} Per-class evaluation results: \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        for i in range(len(class_acc)):
            eval_info += 'Class {} (train counts {}): {:.3f} \n'.format(i, self.train_class_counts[loader_uni_class][i],
                                                                        class_acc[i] * 100)

        # Record missing classes in evaluation sets if exist
        missing_classes = list(set(loader.dataset.class_labels.values()) - set(loader_uni_class))
        eval_info += 'Missing classes in evaluation set: '
        for c in missing_classes:
            eval_info += 'Class {} (train counts {})'.format(c, self.train_class_counts[c])

        return eval_info, class_acc.mean(), overall_acc

    def save_model(self):
        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path))
        self.net.save(self.weights_path)
