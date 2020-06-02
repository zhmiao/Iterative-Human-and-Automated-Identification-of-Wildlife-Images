import os
import numpy as np
import copy
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm, Algorithm, stage_2_metric, acc, WarmupScheduler
from .plain_memory_stage_2_conf_pseu import PlainMemoryStage2_ConfPseu
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model


def load_data(args, conf_preds, pseudo_labels=None):

    """
    Dataloading function. This function can change alg by alg as well.
    """

    print('Using class indices: {} \n'.format(class_indices[args.class_indices]))

    cls_idx = class_indices[args.class_indices]

    trainloader_no_up = load_dataset(name=args.dataset_name,
                                     class_indices=cls_idx,
                                     dset='train',
                                     transform='eval',
                                     split=None,
                                     rootdir=args.dataset_root,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     cas_sampler=False,
                                     conf_preds=conf_preds,
                                     pseudo_labels=pseudo_labels,
                                     GTPS_mode=None)

    # For the first couple of epochs, no sampler, no oltr, set shuffle to True, set cas sampler to False
    trainloader_up_gt = load_dataset(name=args.dataset_name,
                                     class_indices=cls_idx,
                                     dset='train',
                                     transform='train_strong',
                                     split=None,
                                     rootdir=args.dataset_root,
                                     batch_size=int(args.batch_size / 2),
                                     shuffle=True,  # Here
                                     num_workers=args.num_workers,
                                     cas_sampler=False,  # Here
                                     conf_preds=conf_preds,
                                     pseudo_labels=None,
                                     GTPS_mode='GT',
                                     blur=True)

    trainloader_up_ps = load_dataset(name=args.dataset_name,
                                         class_indices=cls_idx,
                                         dset='train',
                                         transform='train_strong',
                                         split=None,
                                         rootdir=args.dataset_root,
                                         batch_size=int(args.batch_size / 2),
                                         shuffle=True,  # Here
                                         num_workers=args.num_workers,
                                         cas_sampler=False,  # Here
                                         conf_preds=conf_preds,
                                         pseudo_labels=pseudo_labels,
                                         GTPS_mode='PS',
                                         blur=True)

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
                              GTPS_mode=None)

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
                             GTPS_mode=None)

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

    return trainloader_no_up, trainloader_up_gt, trainloader_up_ps, testloader, valloader, deployloader


@register_algorithm('GTPSMemoryStage2_ConfPseu')
class GTPSMemoryStage2_ConfPseu(PlainMemoryStage2_ConfPseu):

    """
    Overall training function.
    """

    name = 'GTPSMemoryStage2_ConfPseu'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        self.args = args
        self.logger = self.args.logger
        self.weights_path = './weights/{}/{}_{}.pth'.format(self.args.algorithm, self.args.conf_id, self.args.session)

        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)

        # Training epochs and logging intervals
        self.log_interval = args.log_interval
        self.conf_preds = list(np.fromfile(args.weights_init.replace('.pth', '_conf_preds.npy')).astype(int))
        self.stage_1_mem_flat = np.fromfile(args.weights_init.replace('.pth', '_centroids.npy'), dtype=np.float32)
        self.pseudo_labels = list(np.fromfile(args.weights_init.replace('.pth', '_init_pseudo.npy'), dtype=np.int))

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader_no_up, self.trainloader_up_gt, self.trainloader_up_ps, self.testloader, \
            self.valloader, self.deployloader = load_data(args, self.conf_preds, pseudo_labels=self.pseudo_labels)
        self.train_unique_labels, self.train_class_counts = self.trainloader_no_up.dataset.class_counts_cal()
        self.train_annotation_counts = self.trainloader_no_up.dataset.class_counts_cal_ann()

        if args.limit_steps:
            self.logger.info('** LIMITING STEPS!!! **')
            self.max_batch = len(self.trainloader_no_up)
        else:
            self.max_batch = None

    def reset_trainloader(self):

        self.logger.info('\nReseting training loader and sampler with pseudo labels.')

        cls_idx = class_indices[self.args.class_indices]

        self.trainloader_up_gt = load_dataset(name=self.args.dataset_name,
                                         class_indices=cls_idx,
                                         dset='train',
                                         transform='train_strong',
                                         split=None,
                                         rootdir=self.args.dataset_root,
                                         batch_size=int(self.args.batch_size / 2),
                                         shuffle=False,  # Here
                                         num_workers=self.args.num_workers,
                                         cas_sampler=True,  # Here
                                         conf_preds=self.conf_preds,
                                         pseudo_labels=None,
                                         GTPS_mode='GT',
                                         blur=True)

        self.trainloader_up_ps = load_dataset(name=self.args.dataset_name,
                                             class_indices=cls_idx,
                                             dset='train',
                                             transform='train_strong',
                                             split=None,
                                             rootdir=self.args.dataset_root,
                                             batch_size=int(self.args.batch_size / 2),
                                             shuffle=False,  # Here
                                             num_workers=self.args.num_workers,
                                             cas_sampler=True,  # Here
                                             conf_preds=self.conf_preds,
                                             pseudo_labels=self.pseudo_labels,
                                             GTPS_mode='PS',
                                             blur=True)

    def train_warm_epoch(self, epoch):

        self.net.feature.train()
        self.net.fc_hallucinator.train()

        iter_gt = iter(self.trainloader_up_gt)
        iter_ps = iter(self.trainloader_up_ps)

        if self.max_batch is not None and self.max_batch < len(self.trainloader_up_ps):
            N = self.max_batch
        else:
            N = len(self.trainloader_up_ps)

        for batch_idx in range(N):

            # log basic adda train info
            info_str = '[Warm up training for hallucination (Stage 2)] '
            info_str += 'Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                              N, 100 * batch_idx / N)

            try:
                data_gt, labels_gt, _, _ = next(iter_gt)
            except StopIteration:
                iter_gt = iter(self.trainloader_up_gt)
                data_gt, labels_gt, _, _ = next(iter_gt)

            data_ps, labels_ps, _, _ = next(iter_ps)

            data = torch.cat((data_gt, data_ps), dim=0)
            labels = torch.cat((labels_gt, labels_ps), dim=0)

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
            logits = self.net.fc_hallucinator(feats)

            # TODO: Make sure halluciniator is properly initialized!!!!!

            # calculate loss
            loss = self.net.criterion_cls(logits, labels)

            #############################
            # Backward and optimization #
            #############################
            # zero gradients for optimizer
            self.opt_feats.zero_grad()
            self.opt_fc_hall.zero_grad()
            # loss backpropagation
            loss.backward()
            # optimize step
            self.opt_feats.step()
            self.opt_fc_hall.step()

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

        self.sch_feats.step()
        self.sch_fc_hall.step()

    def train_memory_epoch(self, epoch):

        self.net.feature.train()
        self.net.fc_hallucinator.train()
        self.net.fc_selector.train()
        self.net.cosnorm_classifier.train()
        self.net.criterion_ctr.train()

        iter_gt = iter(self.trainloader_up_gt)
        iter_ps = iter(self.trainloader_up_ps)

        # if self.max_batch is not None and (self.max_batch) * 2 < len(self.trainloader_up_ps):
        #     N = self.max_batch * 2
        # else:
        #     N = len(self.trainloader_up_ps)

        if self.max_batch is not None:
            N = self.max_batch * 2
        else:
            N = len(self.trainloader_up_ps)

        for batch_idx in range(N):

            # log basic adda train info
            info_str = '[Memory training (Stage 2)] '
            info_str += 'Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                              N, 100 * batch_idx / N)

            try:
                data_gt, labels_gt, _, _ = next(iter_gt)
            except StopIteration:
                iter_gt = iter(self.trainloader_up_gt)
                data_gt, labels_gt, _, _ = next(iter_gt)

            data_ps, labels_ps, _, _ = next(iter_ps)

            data = torch.cat((data_gt, data_ps), dim=0)
            labels = torch.cat((labels_gt, labels_ps), dim=0)

            ########################
            # Setup data variables #
            ########################
            # assign devices
            data, labels = data.cuda(), labels.cuda()
            data.requires_grad = False
            labels.requires_grad = False

            ####################
            # Forward and loss #
            ####################

            feats, logits, _ = self.memory_forward(data)

            preds = logits.argmax(dim=1)

            # calculate loss
            xent_loss = self.net.criterion_cls(logits, labels)
            ctr_loss = self.net.criterion_ctr(feats.clone(), labels)
            loss = xent_loss + self.args.ctr_loss_weight * ctr_loss

            #############################
            # Backward and optimization #
            #############################
            # zero gradients for optimizer
            self.opt_feats.zero_grad()
            self.opt_fc_hall.zero_grad()
            self.opt_fc_sel.zero_grad()
            self.opt_cos_clf.zero_grad()
            self.opt_mem.zero_grad()
            # loss backpropagation
            loss.backward()
            # optimize step
            self.opt_feats.step()
            self.opt_fc_hall.step()
            self.opt_fc_sel.step()
            self.opt_cos_clf.step()
            self.opt_mem.step()

            ###########
            # Logging #
            ###########
            if batch_idx % self.log_interval == 0:
                # compute overall acc
                acc = (preds == labels).float().mean()
                # log update info
                info_str += 'Acc: {:0.1f} Xent: {:.3f}'.format(acc.item() * 100, loss.item())
                self.logger.info(info_str)

        self.sch_feats.step()
        self.sch_fc_hall.step()
        self.sch_fc_sel.step()
        self.sch_cos_clf.step()
        self.sch_mem.step()

    def evaluate_epoch(self, loader, hall=False):

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
                if hall:
                    feats = self.net.feature(data)
                    logits = self.net.fc_hallucinator(feats)
                else:
                    _, logits, values_nn = self.memory_forward(data)
                    # scale logits with reachability
                    # reachability_logits = (40 / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                    reachability_logits = (18 / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                    logits = reachability_logits * logits

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
                eval_info += 'Class {} (train counts {} / '.format(i, self.train_class_counts[i])
                eval_info += 'ann counts {}): '.format(self.train_annotation_counts[i])
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

    def deploy_epoch(self, loader):

        self.net.eval()

        total_file_id = []
        total_preds = []
        total_max_probs = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for data, file_id in tqdm(loader, total=len(loader)):

                # setup data
                data = data.cuda()
                data.requires_grad = False

                # forward
                _, logits, values_nn = self.memory_forward(data)

                # scale logits with reachability
                reachability_logits = (18 / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                logits = reachability_logits * logits

                # compute correct
                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                total_preds.append(preds.detach().cpu().numpy())
                total_max_probs.append(max_probs.detach().cpu().numpy())
                total_file_id.append(file_id)

        total_file_id = np.concatenate(total_file_id, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)
        total_max_probs = np.concatenate(total_max_probs, axis=0)

        eval_info = '{} Picking confident samples... \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

        conf_preds = np.zeros(len(total_preds))
        conf_preds[total_max_probs > self.args.theta] = 1

        total_file_id_conf = total_file_id[conf_preds == 1]
        total_preds_conf = total_preds[conf_preds == 1]

        total_file_id_unconf = total_file_id[conf_preds == 0]
        total_preds_unconf = total_preds[conf_preds == 0]

        eval_info += 'Total confident sample count is {} out of {} non-empty samples ({:3f}%) \n'.format(len(total_preds_conf), len(total_preds), 100 * (len(total_preds_conf) / len(total_preds)))
        eval_info += 'Total unconfident sample count is {} out of {} non-empty samples ({:3f}%) \n'.format(len(total_preds_unconf), len(total_preds), 100 * (len(total_preds_unconf) / len(total_preds)))

        return eval_info, (total_file_id_conf, total_preds_conf), (total_file_id_unconf, total_preds_unconf)
