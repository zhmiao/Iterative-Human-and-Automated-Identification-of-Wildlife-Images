import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm, Algorithm, acc, ood_metric
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model


def load_data(args):

    """
    Dataloading function. This function can change alg by alg as well.
    """

    print('Using class indices: {} \n'.format(class_indices[args.class_indices]))

    cls_idx = class_indices[args.class_indices]

    trainloader = load_dataset(name=args.dataset_name,
                               class_indices=cls_idx,
                               dset='train',
                               transform=args.train_transform,
                               rootdir=args.dataset_root,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               cas_sampler=False)

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
                                dset='train',
                                transform='eval',
                                rootdir=args.dataset_root,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                cas_sampler=False)

    return trainloader, valloader, valloaderunknown, deployloader


@register_algorithm('PlainStage1')
class PlainStage1(Algorithm):

    """
    Overall training function.
    """

    name = 'PlainStage1'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(PlainStage1, self).__init__(args=args)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader, self.valloader,\
        self.valloaderunknown, self.deployloader = load_data(args)
        _, self.train_class_counts = self.trainloader.dataset.class_counts_cal()
        self.train_annotation_counts = self.train_class_counts

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init, num_layers=self.args.num_layers, init_feat_only=True)

        self.set_optimizers()

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.logger.info('\nLoading from {}'.format(self.weights_path))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.weights_path, num_layers=self.args.num_layers, init_feat_only=False)

    def set_optimizers(self):
        self.logger.info('** SETTING OPTIMIZERS!!! **')
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

    def train(self):

        best_epoch = 0
        best_acc = 0.

        for epoch in range(self.num_epochs):

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

    def evaluate(self, loader, ood=False):
        if ood:
            eval_info, f1, _ = self.ood_evaluate_epoch(loader, self.valloaderunknown)
            self.logger.info(eval_info)
            return f1
        else:
            eval_info, eval_acc_mac, eval_acc_mic = self.evaluate_epoch(loader)
            self.logger.info(eval_info)
            return eval_acc_mac, eval_acc_mic

    def deploy(self, loader):
        eval_info, f1, conf_preds = self.deploy_epoch(loader)
        self.logger.info(eval_info)
        conf_preds_path = self.weights_path.replace('.pth', '_conf_preds.npy')
        self.logger.info('Saving confident predictions to {}'.format(conf_preds_path))
        conf_preds.tofile(conf_preds_path)
        return f1

    def train_epoch(self, epoch):

        self.net.train()

        N = len(self.trainloader)

        for batch_idx, (data, labels) in enumerate(self.trainloader):

            # log basic adda train info
            info_str = '[Training {} - Stage 1] Epoch: {} [{}/{} ({:.2f}%)] '.format(self.net.name, epoch, batch_idx,
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

    def evaluate_epoch(self, loader):
        self.net.eval()
        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()
        total_preds, total_labels, _ = self.evaluate_forward(loader, ood=False)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        eval_info, mac_acc, mic_acc = self.evaluate_metric(total_preds, total_labels, 
                                                  eval_class_counts, ood=False)
        return eval_info, mac_acc, mic_acc

    def ood_evaluate_epoch(self, loader_in, loader_out):
        self.net.eval()
        # Get unique classes in the loader and corresponding counts
        loader_uni_class_in, eval_class_counts_in = loader_in.dataset.class_counts_cal()
        loader_uni_class_out, eval_class_counts_out = loader_out.dataset.class_counts_cal()

        self.logger.info("Forward through in test loader\n")
        total_preds_in, total_labels_in, _ = self.evaluate_forward(loader_in, ood=True)
        total_preds_in = np.concatenate(total_preds_in, axis=0)
        total_labels_in = np.concatenate(total_labels_in, axis=0)

        self.logger.info("Forward through out test loader\n")
        total_preds_out, total_labels_out, _ = self.evaluate_forward(loader_out, ood=True)
        total_preds_out = np.concatenate(total_preds_out, axis=0)
        total_labels_out = np.concatenate(total_labels_out, axis=0)

        total_preds = np.concatenate((total_preds_out, total_preds_in), axis=0)
        total_labels = np.concatenate((total_labels_out, total_labels_in), axis=0)
        loader_uni_class = np.concatenate((loader_uni_class_out, loader_uni_class_in), axis=0)
        eval_class_counts = np.concatenate((eval_class_counts_out, eval_class_counts_in), axis=0)

        eval_info, f1, conf_preds = self.evaluate_metric(total_preds, total_labels, 
                                                         eval_class_counts, ood=True)

        return eval_info, f1, conf_preds

    def deploy_epoch(self, loader):
        self.net.eval()
        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()
        total_preds, total_labels, _ = self.evaluate_forward(loader, ood=True)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        eval_info, f1, conf_preds = self.evaluate_metric(total_preds, total_labels, 
                                                         eval_class_counts, ood=True)
        return eval_info, f1, conf_preds

    def evaluate_forward(self, loader, ood=False):
        total_preds = []
        total_labels = []
        total_logits = []

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

                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                # Set unconfident prediction to -1
                if ood:
                    preds[max_probs < self.args.theta] = -1

                total_preds.append(preds.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())
                total_logits.append(logits.detach().cpu().numpy())

        return total_preds, total_labels, total_logits

    def evaluate_metric(self, total_preds, total_labels, eval_class_counts, ood=False):
        if ood:
            f1,\
            class_acc_confident, class_percent_confident, false_pos_percent,\
            class_wrong_percent_unconfident,\
            percent_unknown, total_unknown, total_known, conf_preds = ood_metric(total_preds,
                                                                                 total_labels,
                                                                                 eval_class_counts)

            eval_info = '{} Per-class evaluation results: \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

            for i in range(len(class_acc_confident)):
                eval_info += 'Class {} (tr {} / '.format(i, self.train_class_counts[i])
                eval_info += 'ann {}): '.format(self.train_annotation_counts[i])
                eval_info += 'Conf %: {:.2f}; '.format(class_percent_confident[i] * 100)
                eval_info += 'Unconf wrong %: {:.2f}; '.format(class_wrong_percent_unconfident[i] * 100)
                eval_info += 'Conf Acc: {:.3f}; \n'.format(class_acc_confident[i] * 100)

            eval_info += 'Overall F1: {:.3f}; \n'.format(f1)
            eval_info += 'False positive %: {:.3f}; \n'.format(false_pos_percent * 100)
            eval_info += 'Selected unknown %: {:.3f} ({}/{}); \n'.format(percent_unknown * 100,
                                                                        int(percent_unknown * total_unknown),
                                                                        total_unknown)

            eval_info += 'Avg conf %: {:.3f} ({}/{}); \n'.format(class_percent_confident.mean() * 100,
                                                                 int(class_percent_confident.mean() * total_known),
                                                                 total_known)
            eval_info += 'Avg unconf wrong %: {:.3f}; \n'.format(class_wrong_percent_unconfident.mean() * 100)
            eval_info += 'Conf acc %: {:.3f}; \n'.format(class_acc_confident.mean() * 100)

            return eval_info, f1, conf_preds
        else:
            class_acc, mac_acc, mic_acc = acc(total_preds, total_labels)

            eval_info = '{} Per-class evaluation results: \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            for i in range(len(class_acc)):
                eval_info += 'Class {} (tr {} / '.format(i, self.train_class_counts[i])
                eval_info += 'ann {}): '.format(self.train_annotation_counts[i])
                eval_info += 'Acc {:.3f} \n'.format(class_acc[i] * 100)

            eval_info += 'Macro Acc: {:.3f}; Micro Acc: {:.3f}\n'.format(mac_acc * 100, mic_acc * 100)
            return eval_info, mac_acc, mic_acc

    def save_model(self):
        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path))
        self.net.save(self.weights_path)