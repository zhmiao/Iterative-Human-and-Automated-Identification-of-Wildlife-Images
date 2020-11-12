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

    trainloader_no_up = load_dataset(name=args.dataset_name,
                                     class_indices=cls_idx,
                                     dset='train',
                                     transform='eval',
                                     rootdir=args.dataset_root,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     cas_sampler=False,
                                     conf_preds=conf_preds,
                                     pseudo_labels_hard=None,
                                     pseudo_labels_soft=None,
                                     GTPS_mode=None)

    trainloader_up = load_dataset(name=args.dataset_name,
                                  class_indices=cls_idx,
                                  dset='train',
                                  transform='train',
                                  rootdir=args.dataset_root,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  cas_sampler=False,
                                  conf_preds=conf_preds,
                                  pseudo_labels_hard=None,
                                  pseudo_labels_soft=None,
                                  GTPS_mode='GT')

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

    return trainloader_no_up, trainloader_up, valloader, valloaderunknown, deployloader


@register_algorithm('MemoryStage2_ConfPseu')
class MemoryStage2_ConfPseu(Algorithm):

    """
    Overall training function.
    """

    name = 'MemoryStage2_ConfPseu'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(MemoryStage2_ConfPseu, self).__init__(args=args)

        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval
        self.conf_preds = list(np.fromfile(args.weights_init.replace('_ft.pth', '_conf_preds.npy')).astype(int))
        self.pseudo_labels = list(np.fromfile(args.weights_init.replace('_ft.pth', '_init_pseudo_hard.npy'), dtype=np.int))

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader_no_up, self.trainloader_up, \
        self.valloader, self.deployloader = load_data(args, self.conf_preds, unconf_only=False,
                                                      pseudo_labels=None)

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

        # Initial centroids
        initial_centroids_path = self.weights_path.replace('.pth', '_init_centroids.npy')
        if os.path.exists(initial_centroids_path):
            self.logger.info('Loading initial centroids from {}.\n'.format(initial_centroids_path))
            initial_centroids = np.fromfile(initial_centroids_path, dtype=np.float32).reshape(-1, self.net.feature_dim)
        else:
            self.logger.info('\nCalculating initial centroids for all stage 2 classes.')
            stage_1_memory = self.stage_1_mem_flat.reshape(-1, self.net.feature_dim)
            initial_centroids = self.centroids_cal(self.trainloader_no_up, use_pseudo=True).clone().detach().cpu().numpy()
            initial_centroids[:len(stage_1_memory)] = stage_1_memory
            initial_centroids.tofile(initial_centroids_path)
            self.logger.info('\nInitial centroids saved to {}.'.format(initial_centroids_path))

        # TODO: Check if ground truth labels have all classes included!!!!!!!!!!

        # Intitialize centroids using named parameter to avoid possible bugs
        with torch.no_grad():
            for name, param in self.net.criterion_ctr.named_parameters():
                if name == 'centroids':
                    print('\nPopulating initial centroids.\n')
                    param.copy_(torch.from_numpy(initial_centroids))

        ######################
        # Optimization setup #
        ######################
        # Setup optimizer and optimizer scheduler
        self.opt_feats = optim.SGD([{'params': self.net.feature.parameters(),
                                     'lr': self.args.lr_feature,
                                     'momentum': self.args.momentum_feature,
                                     'weight_decay': self.args.weight_decay_feature}])
        self.sch_feats = WarmupScheduler(self.opt_feats, decay1=self.args.decay1, decay2=self.args.decay2,
                                         gamma=self.args.gamma, len_epoch=len(self.trainloader_no_up),
                                         warmup_epochs=self.args.sch_warmup)

        self.opt_fc_hall = optim.SGD([{'params': self.net.fc_hallucinator.parameters(),
                                       'lr': self.args.lr_classifier,
                                       'momentum': self.args.momentum_classifier,
                                       'weight_decay': self.args.weight_decay_classifier}])
        self.sch_fc_hall = WarmupScheduler(self.opt_fc_hall, decay1=self.args.decay1, decay2=self.args.decay2,
                                           gamma=self.args.gamma, len_epoch=len(self.trainloader_no_up),
                                           warmup_epochs=self.args.sch_warmup)

        self.opt_fc_sel = optim.SGD([{'params': self.net.fc_selector.parameters(),
                                      'lr': self.args.lr_classifier,
                                      'momentum': self.args.momentum_classifier,
                                      'weight_decay': self.args.weight_decay_classifier}])
        self.sch_fc_sel = WarmupScheduler(self.opt_fc_sel, decay1=self.args.decay1, decay2=self.args.decay2,
                                          gamma=self.args.gamma, len_epoch=len(self.trainloader_no_up),
                                          warmup_epochs=self.args.sch_warmup)

        self.opt_cos_clf = optim.SGD([{'params': self.net.cosnorm_classifier.parameters(),
                                       'lr': self.args.lr_classifier,
                                       'momentum': self.args.momentum_classifier,
                                       'weight_decay': self.args.weight_decay_classifier}])
        self.sch_cos_clf = WarmupScheduler(self.opt_cos_clf, decay1=self.args.decay1, decay2=self.args.decay2,
                                           gamma=self.args.gamma, len_epoch=len(self.trainloader_no_up),
                                           warmup_epochs=self.args.sch_warmup)

        self.opt_mem = optim.SGD([{'params': self.net.criterion_ctr.parameters(),
                                   'lr': self.args.lr_memory,
                                   'momentum': self.args.momentum_memory,
                                   'weight_decay': self.args.weight_decay_memory}])
        self.sch_mem = WarmupScheduler(self.opt_mem, decay1=self.args.decay1, decay2=self.args.decay2,
                                       gamma=self.args.gamma, len_epoch=len(self.trainloader_no_up),
                                       warmup_epochs=self.args.sch_warmup)

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.logger.info('\nLoading from {}'.format(self.weights_path))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.weights_path, num_layers=self.args.num_layers, init_feat_only=False)

    def memory_forward(self, data):
        # feature
        feats = self.net.feature(data)

        batch_size = feats.size(0)
        feat_size = feats.size(1)

        # get current centroids and detach it from graph
        centroids = self.net.criterion_ctr.centroids.clone().detach()
        centroids.requires_grad = False

        # set up visual memory
        feats_expand = feats.clone().unsqueeze(1).expand(-1, self.net.num_cls, -1)
        centroids_expand = centroids.clone().unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids.clone()

        # computing reachability
        dist_cur = torch.norm(feats_expand - centroids_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1)

        reachability = (self.args.reachability_scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

        # computing memory feature by querying and associating visual memory
        values_memory = self.net.fc_hallucinator(feats.clone())
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.net.fc_selector(feats.clone())
        concept_selector = concept_selector.tanh()

        # computing meta embedding
        meta_feats = reachability * (feats + concept_selector * memory_feature)

        # final logits
        logits = self.net.cosnorm_classifier(meta_feats)

        return feats, logits, values_nn

    def train_warm_epoch(self, epoch):

        self.net.feature.train()
        self.net.fc_hallucinator.train()

        # N = len(self.trainloader_up)
        if self.max_batch is not None:
            N = self.max_batch
        else:
            N = len(self.trainloader_up)

        for batch_idx, (data, labels, confs, indices) in enumerate(self.trainloader_up):

            if batch_idx > N:
                break

            # log basic adda train info
            info_str = '[Warm up training for hallucination (Stage 2)] '
            info_str += 'Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                              N, 100 * batch_idx / N)

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
            # forward
            feats = self.net.feature(data)
            logits = self.net.fc_hallucinator(feats)

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

    def train_memory_epoch(self, epoch):

        self.sch_feats.step()
        self.sch_fc_hall.step()
        self.sch_fc_sel.step()
        self.sch_cos_clf.step()
        self.sch_mem.step()

        self.net.feature.train()
        self.net.fc_hallucinator.train()
        self.net.fc_selector.train()
        self.net.cosnorm_classifier.train()
        self.net.criterion_ctr.train()

        # N = len(self.trainloader_up)
        if self.max_batch is not None:
            N = self.max_batch
            if len(self.trainloader_up) < N:
                N = len(self.trainloader_up)
        else:
            N = len(self.trainloader_up)

        for batch_idx, (data, labels, confs, indices) in enumerate(self.trainloader_up):

            if batch_idx > N:
                break

            # log basic adda train info
            info_str = '[Memory training (Stage 2)] '
            info_str += 'Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                              N, 100 * batch_idx / N)

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

    def train(self):

        for epoch in range(self.args.hall_warm_up_epochs):
            # Training
            self.train_warm_epoch(epoch)

        best_epoch = 0
        best_acc = 0.

        for epoch in range(self.num_epochs):

            # Training
            self.train_memory_epoch(epoch)

            # Validation
            self.logger.info('\nValidation.')
            val_acc_mac = self.evaluate(self.valloader)
            if val_acc_mac > best_acc:
                self.logger.info('\nUpdating Best Model Weights!!')
                self.net.update_best()
                best_acc = val_acc_mac
                best_epoch = epoch

            # TODO!!!
            # if epoch >= self.args.mem_warm_up_epochs:
            #     self.logger.info('\nGenerating new pseudo labels.')
            #     self.pseudo_labels = self.evaluate_epoch(self.trainloader_no_up, pseudo_label_gen=True)
            #     self.reset_trainloader()

        self.logger.info('\nBest Model Appears at Epoch {}...'.format(best_epoch))
        self.save_model()

    def evaluate_epoch(self, loader, pseudo_label_gen=False):

        self.net.eval()

        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()

        total_prob_diffs = []
        total_preds = []
        total_max_probs = []
        total_labels = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for batch in tqdm(loader, total=len(loader)):

                if loader == self.trainloader_no_up:
                    data, labels, _, _ = batch
                else:
                    data, labels = batch

                # setup data
                data, labels = data.cuda(), labels.cuda()
                data.requires_grad = False
                labels.requires_grad = False

                _, logits, values_nn = self.memory_forward(data)

                # scale logits with reachability
                reachability_logits = (self.args.reachability_scale / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                logits = reachability_logits * logits

                # calculate probs
                probs = F.softmax(logits, dim=1)
                max_probs, preds = probs.max(dim=1)

                # calculate top2 prob differences
                top_2_probs, _ = torch.topk(probs, 2, dim=1)
                prob_diffs = top_2_probs[:, 0] - top_2_probs[:, 1]

                total_prob_diffs.append(prob_diffs.detach().cpu().numpy())
                total_preds.append(preds.detach().cpu().numpy())
                total_max_probs.append(max_probs.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())

        if pseudo_label_gen:

            pseudo_labels = np.concatenate(total_preds, axis=0)

            # Set unconfident pseudo labels to -1
            pseudo_labels[np.concatenate(total_max_probs, axis=0) < self.args.theta] = -1

            return list(pseudo_labels)

        else:

            class_wrong_percent_unconfident, \
            class_correct_percent_unconfident, \
            class_acc_confident, total_unconf, \
            missing_cls_in_test, \
            missing_cls_in_train = stage_2_metric(np.concatenate(total_preds, axis=0),
                                                  np.concatenate(total_max_probs, axis=0),
                                                  np.concatenate(total_labels, axis=0),
                                                  self.train_unique_labels,
                                                  self.args.theta)

            # class_wrong_percent_unconfident, \
            # class_correct_percent_unconfident, \
            # class_acc_confident, total_unconf, \
            # missing_cls_in_test, \
            # missing_cls_in_train = stage_2_metric(np.concatenate(total_preds, axis=0),
            #                                       np.concatenate(total_prob_diffs, axis=0),
            #                                       np.concatenate(total_labels, axis=0),
            #                                       self.train_unique_labels,
            #                                       self.args.theta)

            # Record per class accuracies
            class_acc, mac_acc, mic_acc = acc(np.concatenate(total_preds, axis=0),
                                              np.concatenate(total_labels, axis=0),
                                              self.train_class_counts)

            eval_info = '{} Per-class evaluation results: \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

            for i in range(len(self.train_unique_labels)):
                if i not in missing_cls_in_test:
                    eval_info += 'Class {} (train counts {} '.format(i, self.train_class_counts[i])
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

