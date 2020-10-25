import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm, Algorithm, acc, stage_1_metric
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model
from src.algorithms.plain_stage_1 import load_data, PlainStage1


@register_algorithm('EnergyStage1')
class EnergyStage1(PlainStage1):

    """
    Overall training function.
    """

    name = 'EnergyStage1'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(EnergyStage1, self).__init__(args=args)

        self.trainloaderunknown = load_dataset(name='MOZ_UNKNOWN',
                                               class_indices=class_indices[self.args.class_indices],
                                               dset='train',
                                               transform=self.args.train_transform,
                                               rootdir=self.args.dataset_root,
                                               batch_size=self.args.batch_size * 2,
                                               shuffle=True,
                                               num_workers=self.args.num_workers,
                                               cas_sampler=False)

        self.valloaderunknown = load_dataset(name='MOZ_UNKNOWN',
                                             class_indices=class_indices[self.args.class_indices],
                                             dset='val',
                                             transform='eval',
                                             rootdir=self.args.dataset_root,
                                             batch_size=self.args.batch_size,
                                             shuffle=False,
                                             num_workers=self.args.num_workers,
                                             cas_sampler=False)

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.logger.info('\nLoading from {}'.format(self.weights_path.replace('.pth', '_ft.pth')))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.weights_path.replace('.pth', '_ft.pth'), num_layers=self.args.num_layers, init_feat_only=False)

    def ood_ft_epoch(self, epoch):

        self.net.train()

        N = len(self.trainloader)

        in_iter = iter(self.trainloader)
        out_iter = iter(self.trainloaderunknown)

        # for batch_idx, (data_in, labels) in enumerate(self.trainloader):
        for batch_idx in range(N):

            data_in, labels = next(in_iter)

            try:
                data_out, _ = next(out_iter)
            except StopIteration:
                out_iter = iter(self.trainloaderunknown)
                data_out, _ = next(out_iter)

            data = torch.cat((data_in, data_out), dim=0)

            # log basic adda train info
            info_str = '[OOD FTing {} - Stage 1] Epoch: {} [{}/{} ({:.2f}%)] '.format(self.net.name, epoch, batch_idx,
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

            # calculate xent loss
            xent = self.net.criterion_cls(logits[:len(data_in)], labels)

            # calculate energy loss
            Ec_out = -torch.logsumexp(logits[len(data_in):], dim=1)
            Ec_in = -torch.logsumexp(logits[:len(data_in)], dim=1)
            m_out = -7.
            m_in = -18.
            ebloss = torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean()
            # ebloss = torch.tensor(0.)

            loss = xent + 0.01 * ebloss

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
                preds = logits[:len(data_in)].argmax(dim=1)
                acc = (preds == labels).float().mean()
                # log update info
                info_str += 'Acc: {:0.1f} Xent: {:.3f}, EB: {:.3f}'.format(acc.item() * 100, xent.item(), ebloss.item())
                self.logger.info(info_str)

    def ood_ft(self):

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

        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos(step / total_steps * np.pi))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt_net,
            lr_lambda=lambda step: cosine_annealing(
                step,
                self.num_epochs * len(self.trainloader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / self.args.lr_feature))

        for epoch in range(self.num_epochs):

            # Training
            self.ood_ft_epoch(epoch)

            # Validation
            self.logger.info('\nValidation.')
            val_acc_mac = self.evaluate(self.valloader)
            self.logger.info('\nOOD.')
            # _ = self.deploy(self.deployloader)
            _ = self.evaluate(self.valloader, ood=True)
            self.logger.info('\nUpdating Best Model Weights!!')
            self.net.update_best()

        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path.replace('.pth', '_ft.pth')))
        self.net.save(self.weights_path.replace('.pth', '_ft.pth'))

    def ood_evaluate_epoch(self, loader_in, loader_out, energy_the, T):

        self.net.eval()

        # Get unique classes in the loader and corresponding counts
        loader_uni_class_in, eval_class_counts_in = loader_in.dataset.class_counts_cal()
        loader_uni_class_out, eval_class_counts_out = loader_out.dataset.class_counts_cal()

        self.logger.info("Forward through in test loader\n")
        total_preds_in, total_labels_in, total_energy_score_in = self.ood_forward(loader_in, energy_the, T)
        total_preds_in = np.concatenate(total_preds_in, axis=0)
        total_labels_in = np.concatenate(total_labels_in, axis=0)

        self.logger.info("Forward through out test loader\n")
        total_preds_out, total_labels_out, total_energy_score_out = self.ood_forward(loader_out, energy_the, T)
        total_preds_out = np.concatenate(total_preds_out, axis=0)
        total_labels_out = np.concatenate(total_labels_out, axis=0)

        total_preds = np.concatenate((total_preds_out, total_preds_in), axis=0)
        total_labels = np.concatenate((total_labels_out, total_labels_in), axis=0)
        loader_uni_class = np.concatenate((loader_uni_class_out, loader_uni_class_in), axis=0)
        eval_class_counts = np.concatenate((eval_class_counts_out, eval_class_counts_in), axis=0)

        eval_info, f1, conf_preds = self.ood_metric(total_preds, total_labels, loader_uni_class, eval_class_counts)

        return eval_info, f1, conf_preds

    def evaluate(self, loader, ood=False):
        if ood:
            eval_info, f1, conf_preds = self.ood_evaluate_epoch(loader, self.valloaderunknown, 13.7, T=1.5)
            self.logger.info(eval_info)
            return f1
        else:
            eval_info, eval_acc_mac = self.evaluate_epoch(loader)
            self.logger.info(eval_info)
            return eval_acc_mac

    def deploy_epoch(self, loader, energy_the, T):
        self.net.eval()
        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()
        total_preds, total_labels, total_energy_score = self.ood_forward(loader, energy_the, T)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        eval_info, f1, conf_preds = self.ood_metric(total_preds, total_labels, loader_uni_class, eval_class_counts)
        return eval_info, f1, conf_preds

    def deploy(self, loader):
        eval_info, f1, conf_preds = self.deploy_epoch(loader, 13.7, T=1.5)
        self.logger.info(eval_info)
        # for the in range(10, 16, 1):
        #     self.logger.info('\nThe: {}.'.format(the))
        #     eval_info, f1, conf_preds = self.deploy_epoch(loader, the, T=1.5)
        #     self.logger.info(eval_info)
        # conf_preds_path = self.weights_path.replace('.pth', '_conf_preds.npy')
        # self.logger.info('Saving confident predictions to {}'.format(conf_preds_path))
        # conf_preds.tofile(conf_preds_path)
        return f1

    def ood_forward(self, loader, energy_the, T):
        total_preds = []
        total_labels = []
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

                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                # T = 1.0
                energy_score = -(T * torch.logsumexp(logits / T, dim=1))

                # Set unconfident prediction to -1
                # preds[max_probs < self.args.theta] = -1
                preds[-energy_score <= energy_the] = -1

                total_preds.append(preds.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())
                total_energy_score.append(energy_score.detach().cpu().numpy())

        return total_preds, total_labels, total_energy_score

    def ood_metric(self, total_preds, total_labels, loader_uni_class, eval_class_counts):
        f1,\
        class_acc_confident, class_percent_confident, false_pos_percent,\
        class_wrong_percent_unconfident,\
        percent_unknown, conf_preds = stage_1_metric(total_preds,
                                                     total_labels,
                                                     loader_uni_class,
                                                     eval_class_counts)

        eval_info = '{} Per-class evaluation results: \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

        # for i in range(len(class_acc_confident)):
        #     # TODO
        #     eval_info += 'Class {} (train counts {}):'.format(i, self.train_class_counts[loader_uni_class[loader_uni_class != -1]][i])
        #     eval_info += 'Confident percentage: {:.2f}; '.format(class_percent_confident[i] * 100)
        #     eval_info += 'Unconfident wrong %: {:.2f}; '.format(class_wrong_percent_unconfident[i] * 100)
        #     eval_info += 'Accuracy: {:.3f} \n'.format(class_acc_confident[i] * 100)

        eval_info += 'Overall F1: {:.3f} \n'.format(f1)
        eval_info += 'False positive percentage: {:.3f} \n'.format(false_pos_percent * 100)
        eval_info += 'Selected unknown percentage: {:.3f} \n'.format(percent_unknown * 100)

        eval_info += 'Avg conf %: {:.3f}; \n'.format(class_percent_confident.mean() * 100)
        eval_info += 'Avg unconf wrong %: {:.3f}; \n'.format(class_wrong_percent_unconfident.mean() * 100)
        eval_info += 'Conf acc %: {:.3f}\n'.format(class_acc_confident.mean() * 100)

        return eval_info, f1, conf_preds

