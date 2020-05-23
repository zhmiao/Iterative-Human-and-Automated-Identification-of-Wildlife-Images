import os
import numpy as np
import copy
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm, Algorithm, stage_2_metric, acc, WarmupScheduler
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model


def load_data(args, conf_preds, unconf_only=False, pseudo_labels=None):

    """
    Dataloading function. This function can change alg by alg as well.
    """

    print('Using class indices: {} \n'.format(class_indices[args.class_indices]))

    cls_idx = class_indices[args.class_indices]

    trainloader = load_dataset(name=args.dataset_name,
                               class_indices=cls_idx,
                               dset='train',
                               transform='train',
                               split=args.train_split,
                               rootdir=args.dataset_root,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               cas_sampler=False,
                               conf_preds=conf_preds,
                               pseudo_labels=pseudo_labels,
                               unconf_only=unconf_only)

    testloader = load_dataset(name=args.dataset_name,
                              class_indices=cls_idx,
                              dset='test',
                              transform='eval',
                              split=None,
                              rootdir=args.dataset_root,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              cas_sampler=False,
                              conf_preds=None,
                              unconf_only=False)

    valloader = load_dataset(name=args.dataset_name,
                             class_indices=cls_idx,
                             dset='val',
                             transform='eval',
                             split=None,
                             rootdir=args.dataset_root,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             cas_sampler=False,
                             conf_preds=None,
                             unconf_only=False)

    deployloader = load_dataset(name=args.deploy_dataset_name,
                                class_indices=cls_idx,
                                dset='deploy',
                                transform='eval',
                                split=None,
                                rootdir=args.dataset_root,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                cas_sampler=False)

    return trainloader, testloader, valloader, deployloader


@register_algorithm('PlainStage2_ConfPseu')
class PlainStage2_ConfPseu(Algorithm):

    """
    Overall training function.
    """

    name = 'PlainStage2_ConfPseu'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(PlainStage2_ConfPseu, self).__init__(args=args)

        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval
        self.conf_preds = list(np.fromfile(args.weights_init.replace('.pth', '_conf_preds.npy')).astype(int))
        # self.stage_1_mem_flat = np.fromfile(args.weights_init.replace('.pth', '_centroids.npy'), dtype=np.float32)

        self.pseudo_labels = list(np.fromfile(args.weights_init.replace('.pth', '_init_pseudo.npy'), dtype=np.int))

        # TODO: Check pseudo_labels accuracy!!!

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader, self.testloader, \
        self.valloader, self.deployloader = load_data(args, self.conf_preds, unconf_only=False,
                                                      pseudo_labels=self.pseudo_labels)
        self.train_unique_labels, self.train_class_counts = self.trainloader.dataset.class_counts_cal()
        self.train_annotation_counts = self.trainloader.dataset.class_counts_cal_ann()

    def reset_trainloader(self):

        self.logger.info('\nReseting training loader and sampler with pseudo labels.')

        cls_idx = class_indices[self.args.class_indices]

        self.trainloader_up = load_dataset(name=self.args.dataset_name,
                                           class_indices=cls_idx,
                                           dset='train',
                                           transform='train',
                                           split=None,
                                           rootdir=self.args.dataset_root,
                                           batch_size=self.args.batch_size,
                                           shuffle=False,
                                           num_workers=self.args.num_workers,
                                           cas_sampler=True,
                                           conf_preds=self.conf_preds,
                                           pseudo_labels=self.pseudo_labels,
                                           unconf_only=False)

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        # The only thing different from plain resnet is that init_feat_only is set to true now
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init, num_layers=self.args.num_layers, init_feat_only=True)

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

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.logger.info('\nLoading from {}'.format(self.weights_path))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.weights_path, num_layers=self.args.num_layers, init_feat_only=False)

    def train_epoch(self, epoch):

        self.net.train()

        N = len(self.trainloader)

        for batch_idx, (data, labels, _, _) in enumerate(self.trainloader):

            # log basic adda train info
            info_str = '[Train plain] Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
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

        self.scheduler.step()

    def train(self):

        best_epoch = 0
        best_acc = 0.

        for epoch in range(self.num_epochs):

            # Training
            self.train_epoch(epoch)

            # Validation
            self.logger.info('\nValidation.')
            val_acc_mac = self.evaluate(self.valloader)
            if val_acc_mac > best_acc:
                self.logger.info('\nUpdating Best Model Weights!!')
                self.net.update_best()
                best_acc = val_acc_mac
                best_epoch = epoch

            # # TODO!!!
            # if epoch >= self.args.mem_warm_up_epochs:
            #     self.logger.info('\nGenerating new pseudo labels.')
            #     self.pseudo_labels = self.evaluate_epoch(self.trainloader_no_up, pseudo_label_gen=True)
            #     self.reset_trainloader()

        self.logger.info('\nBest Model Appears at Epoch {}...'.format(best_epoch))
        self.save_model()

    def evaluate_epoch(self, loader):

        self.net.eval()

        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()

        total_preds = []
        total_max_probs = []
        total_labels = []

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

                # compute correct
                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                total_preds.append(preds.detach().cpu().numpy())
                total_max_probs.append(max_probs.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())

        class_wrong_percent_unconfident, \
        class_correct_percent_unconfident, \
        class_acc_confident, total_unconf, \
        missing_cls_in_test, \
        missing_cls_in_train = stage_2_metric(np.concatenate(total_preds, axis=0),
                                              np.concatenate(total_max_probs, axis=0),
                                              np.concatenate(total_labels, axis=0),
                                              self.train_unique_labels,
                                              self.args.theta)

        # Record per class accuracies
        class_acc, mac_acc, mic_acc = acc(np.concatenate(total_preds, axis=0),
                                          np.concatenate(total_labels, axis=0),
                                          self.train_class_counts)

        eval_info = '{} Per-class evaluation results: \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

        for i in range(len(self.train_unique_labels)):
            if i not in missing_cls_in_test:
                eval_info += 'Class {} (train counts {}): '.format(i, self.train_class_counts[i])
                eval_info += 'Acc {:.3f} '.format(class_acc[i] * 100)
                eval_info += 'Unconfident wrong % {:.3f} '.format(class_wrong_percent_unconfident[i] * 100)
                eval_info += 'Unconfident correct % {:.3f} '.format(class_correct_percent_unconfident[i] * 100)
                eval_info += 'Confident Acc {:.3f} \n'.format(class_acc_confident[i] * 100)

        eval_info += 'Total unconfident samples: {}\n'.format(total_unconf)
        eval_info += 'Missing classes in test: {}\n'.format(missing_cls_in_test)

        eval_info += 'Macro Acc: {:.3f}; '.format(mac_acc * 100)
        eval_info += 'Micro Acc: {:.3f}; '.format(mic_acc * 100)
        eval_info += 'Avg Unconf Wrong %: {:.3f}; '.format(class_wrong_percent_unconfident.mean() * 100)
        eval_info += 'Avg Unconf Correct %: {:.3f}; '.format(class_correct_percent_unconfident.mean() * 100)
        eval_info += 'Conf cc %: {:.3f}\n'.format(class_acc_confident.mean() * 100)

        # Record missing classes in evaluation sets if exist
        missing_classes = list(set(loader.dataset.class_indices.values()) - set(loader_uni_class))
        eval_info += 'Missing classes in evaluation set: '
        for c in missing_classes:
            eval_info += 'Class {} (train counts {})'.format(c, self.train_class_counts[c])

        return eval_info, class_acc.mean()

    def centroids_cal(self, loader, use_pseudo=False):

        self.net.eval()

        centroids = torch.zeros(len(class_indices[self.args.class_indices]),
                                self.net.feature_dim).cuda()

        with torch.set_grad_enabled(False):

            # TODO: use pseudo labels for initial centroids generation
            for batch in tqdm(loader, total=len(loader)):

                if loader == self.trainloader_no_up:
                    data, labels, confs, indices = batch
                else:
                    data, labels = batch

                # setup data
                data, labels = data.cuda(), labels.cuda()
                data.requires_grad = False
                labels.requires_grad = False
                # forward
                feats = self.net.feature(data)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += feats[i]

        # Get data counts
        _, loader_class_counts = loader.dataset.class_counts_cal()
        # Average summed features with class count
        centroids /= torch.tensor(loader_class_counts).float().unsqueeze(1).cuda()

        return centroids

    def evaluate(self, loader):
        eval_info, eval_acc_mac = self.evaluate_epoch(loader)
        self.logger.info(eval_info)
        return eval_acc_mac

    def deploy_epoch(self):
        pass

    def deploy(self, loader):
        pass

    def save_model(self):
        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path))
        self.net.save(self.weights_path)

