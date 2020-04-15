import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm, Algorithm, stage_2_metric
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.data.class_aware_sampler import ClassAwareSampler
from src.models.utils import get_model


def load_data(args, conf_preds, unknown_only=False):

    """
    Dataloading function. This function can change alg by alg as well.
    """

    print('Using class indices: {} \n'.format(class_indices[args.class_indices]))

    cls_idx = class_indices[args.class_indices]

    trainloader_no_up = load_dataset(name=args.dataset_name,
                                     class_indices=cls_idx,
                                     dset='train',
                                     transform='eval',
                                     split=args.train_split,
                                     rootdir=args.dataset_root,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     sampler=None,
                                     conf_preds=conf_preds,
                                     unknown_only=unknown_only)

    # Use replace S1 to S2 for evaluation
    testloader = load_dataset(name=args.dataset_name,
                              class_indices=cls_idx,
                              dset='test',
                              transform='eval',
                              split=None,
                              rootdir=args.dataset_root,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              sampler=None,
                              conf_preds=None,
                              unknown_only=False)

    # Use replace S1 to S2 for evaluation
    valloader = load_dataset(name=args.dataset_name,
                             class_indices=cls_idx,
                             dset='val',
                             transform='eval',
                             split=None,
                             rootdir=args.dataset_root,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             sampler=None,
                             conf_preds=None,
                             unknown_only=False)

    return trainloader_no_up, testloader, valloader


@register_algorithm('MemoryStage2')
class MemoryStage2(Algorithm):

    """
    Overall training function.
    """

    name = 'MemoryStage2'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(MemoryStage2, self).__init__(args=args)

        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval
        self.conf_preds = list(np.fromfile(args.weights_init.replace('.pth', '_conf_preds.npy')).astype(int))
        self.stage_1_mem_flat = np.fromfile(args.weights_init.replace('.pth', '_centroids.npy'), dtype=np.float32)
        self.pseudo_labels = torch.from_numpy(np.fromfile(args.weights_init.replace('.pth', '_init_pseudo.npy'),
                                                          dtype=np.int))

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader_no_up,\
        self.testloader, self.valloader = load_data(args, self.conf_preds, unknown_only=False)
        self.train_unique_labels, self.train_class_counts = self.trainloader_no_up.dataset.class_counts_cal()
        self.train_annotation_counts = self.trainloader_no_up.dataset.class_counts_cal_ann()

        self.trainloader_up = None

        if args.limit_steps:
            self.logger.info('** LIMITING STEPS!!! **')
            self.max_batch = len(self.trainloader_no_up)
        else:
            self.max_batch = None

    def reset_trainloader(self):
        self.logger.info('\nReseting training loader and sampler with pseudo labels.')
        cls_idx = class_indices[self.args.class_indices]
        sampler = ClassAwareSampler(labels=self.pseudo_labels, num_samples_cls=self.args.num_samples_cls)
        self.trainloader_up = load_dataset(name=self.args.dataset_name,
                                           class_indices=cls_idx,
                                           dset='train',
                                           transform='train',
                                           split=None,
                                           rootdir=self.args.dataset_root,
                                           batch_size=self.args.batch_size,
                                           shuffle=False,
                                           num_workers=self.args.num_workers,
                                           sampler=sampler,
                                           conf_preds=self.conf_preds,
                                           unknown_only=False)

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
            initial_centroids = self.centroids_cal(self.trainloader_no_up).clone().detach().cpu().numpy()
            initial_centroids[:len(stage_1_memory)] = stage_1_memory
            initial_centroids.tofile(initial_centroids_path)
            self.logger.info('\nInitial centroids saved to {}.'.format(initial_centroids_path))

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
        self.sch_feats = optim.lr_scheduler.StepLR(self.opt_feats, step_size=self.args.step_size,
                                                   gamma=self.args.gamma)

        self.opt_fc_hall = optim.SGD([{'params': self.net.fc_hallucinator.parameters(),
                                       'lr': self.args.lr_classifier,
                                       'momentum': self.args.momentum_classifier,
                                       'weight_decay': self.args.weight_decay_classifier}])
        self.sch_fc_hall = optim.lr_scheduler.StepLR(self.opt_fc_hall, step_size=self.args.step_size,
                                                     gamma=self.args.gamma)

        self.opt_fc_sel = optim.SGD([{'params': self.net.fc_selector.parameters(),
                                      'lr': self.args.lr_classifier,
                                      'momentum': self.args.momentum_classifier,
                                      'weight_decay': self.args.weight_decay_classifier}])
        self.sch_fc_sel = optim.lr_scheduler.StepLR(self.opt_fc_sel, step_size=self.args.step_size,
                                                    gamma=self.args.gamma)

        self.opt_cos_clf = optim.SGD([{'params': self.net.cosnorm_classifier.parameters(),
                                       'lr': self.args.lr_classifier,
                                       'momentum': self.args.momentum_classifier,
                                       'weight_decay': self.args.weight_decay_classifier}])
        self.sch_cos_clf = optim.lr_scheduler.StepLR(self.opt_cos_clf, step_size=self.args.step_size,
                                                     gamma=self.args.gamma)

        self.opt_mem = optim.SGD([{'params': self.net.criterion_ctr.parameters(),
                                   'lr': self.args.lr_classifier,
                                   'momentum': self.args.momentum_classifier,
                                   'weight_decay': self.args.weight_decay_classifier}])
        self.sch_mem = optim.lr_scheduler.StepLR(self.opt_mem, step_size=self.args.step_size,
                                                 gamma=self.args.gamma)

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.logger.info('\nLoading from {}'.format(self.weights_path))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.weights_path, num_layers=self.args.num_layers, init_feat_only=False)

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
            # assign pseudo labels using initial pseudo labels
            labels[confs == 1] = self.pseudo_labels[indices][confs == 1]
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

        self.net.feature.train()
        self.net.fc_hallucinator.train()
        self.net.fc_selector.train()
        self.net.cosnorm_classifier.train()
        self.net.criterion_ctr.train()

        # N = len(self.trainloader_up)
        if self.max_batch is not None:
            N = self.max_batch
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
            # assign pseudo labels
            labels[confs == 1] = self.pseudo_labels[indices][confs == 1]
            # assign devices
            data, labels = data.cuda(), labels.cuda()
            data.requires_grad = False
            labels.requires_grad = False

            ####################
            # Forward and loss #
            ####################
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

            # TODO!
            # scale = 10.0
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

    def train(self):

        for epoch in range(self.args.hall_warm_up_epochs):
            # Each epoch, reset training loader and sampler with corresponding pseudo labels.
            self.reset_trainloader()
            # Training
            self.train_warm_epoch(epoch)

        best_acc = 0.

        for epoch in range(self.num_epochs):

            # Each epoch, reset training loader and sampler with corresponding pseudo labels.
            self.reset_trainloader()

            # Training
            self.train_memory_epoch(epoch)

            # Validation
            self.logger.info('\nValidation.')
            val_acc_mac = self.evaluate(self.valloader)
            if val_acc_mac > best_acc:
                self.net.update_best()

            if epoch >= self.args.mem_warm_up_epochs:
                self.logger.info('\nGenerating new pseudo labels.')
                self.pseudo_labels = self.evaluate_epoch(self.trainloader_no_up, pseudo_labels=True)

        self.save_model()

    def evaluate_epoch(self, loader, pseudo_labels=False):

        self.net.eval()

        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()
        loader_class_counts_dict = {loader_uni_class[i]: eval_class_counts[i] for i in range(len(loader_uni_class))}
        eval_class_counts = np.array([loader_class_counts_dict[c]
                                      if c in loader_class_counts_dict else 1e-7 for c in range(len(self.train_unique_labels))])
        class_correct = np.array([0. for _ in range(len(self.train_unique_labels))])

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

                # forward
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

                # TODO!!
                # scale logits with reachability
                # reachability_logits = (self.args.reachability_scale / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                reachability_logits = (15 / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                logits = reachability_logits * logits

                # compute correct
                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)
                for i in range(len(preds)):
                    pred = preds[i]
                    label = labels[i]
                    if pred == label:
                        class_correct[label] += 1

                total_preds.append(preds.detach().cpu().numpy())
                total_max_probs.append(max_probs.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())

        if pseudo_labels:

            return torch.from_numpy(np.concatenate(total_preds, axis=0))

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
            # Record per class accuracies
            # class_acc = class_correct[loader_uni_class] / eval_class_counts[loader_uni_class]
            class_acc = class_correct / eval_class_counts
            overall_acc = class_correct.sum() / eval_class_counts.sum()
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

            # Record missing classes in evaluation sets if exist
            missing_classes = list(set(loader.dataset.class_indices.values()) - set(loader_uni_class))
            eval_info += 'Missing classes in evaluation set: '
            for c in missing_classes:
                eval_info += 'Class {} (train counts {})'.format(c, self.train_class_counts[c])

            return eval_info, class_acc.mean(), overall_acc,\
                   class_wrong_percent_unconfident.mean(),\
                   class_correct_percent_unconfident.mean(),\
                   class_acc_confident.mean()

    def centroids_cal(self, loader):

        self.net.eval()

        centroids = torch.zeros(len(class_indices[self.args.class_indices]),
                                self.net.feature_dim).cuda()

        with torch.set_grad_enabled(False):
            for data, labels, _, _ in tqdm(loader, total=len(loader)):
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
        eval_info, eval_acc_mac, eval_acc_mic,\
        avg_unconf_wrong, avg_unconf_corr, avg_conf_acc = self.evaluate_epoch(loader)

        eval_info += 'Macro Acc: {:.3f}; '.format(eval_acc_mac * 100)
        eval_info += 'Micro Acc: {:.3f}; '.format(eval_acc_mic * 100)
        eval_info += 'Avg Unconf Wrong %: {:.3f}; '.format(avg_unconf_wrong * 100)
        eval_info += 'Avg Unconf Correct %: {:.3f}; '.format(avg_unconf_corr * 100)
        eval_info += 'Conf cc %: {:.3f}\n'.format(avg_conf_acc * 100)

        self.logger.info(eval_info)

        return eval_acc_mac

    def save_model(self):
        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path))
        self.net.save(self.weights_path)

