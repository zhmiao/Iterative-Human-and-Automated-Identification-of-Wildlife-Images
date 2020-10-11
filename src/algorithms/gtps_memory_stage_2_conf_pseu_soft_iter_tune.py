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


def load_data(args, conf_preds):

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
                                     pseudo_labels_hard=None,
                                     pseudo_labels_soft=None,
                                     GTPS_mode=None)

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

    deployloader_ood = load_dataset(name=args.deploy_ood_dataset_name,
                                    class_indices=cls_idx,
                                    dset='deploy',
                                    transform='eval',
                                    split=None,
                                    rootdir=args.dataset_root,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    cas_sampler=False)

    return trainloader_no_up, testloader, valloader, deployloader, deployloader_ood


@register_algorithm('GTPSMemoryStage2_ConfPseu_SoftIter_TUNE')
class GTPSMemoryStage2_ConfPseu_SoftIter_TUNE(PlainMemoryStage2_ConfPseu):

    """
    Overall training function.
    """

    name = 'GTPSMemoryStage2_ConfPseu_SoftIter_TUNE'
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

        self.pseudo_labels_hard = np.fromfile(args.weights_init.replace('.pth', '_init_pseudo_hard.npy'), dtype=np.int)
        self.pseudo_labels_soft = None

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader_no_up, self.testloader, self.valloader, \
        self.deployloader, self.deployloader_ood = load_data(args, self.conf_preds)

        self.train_unique_labels, self.train_class_counts = self.trainloader_no_up.dataset.class_counts_cal()
        self.train_annotation_counts = self.trainloader_no_up.dataset.class_counts_cal_ann()

        if args.limit_steps:
            self.logger.info('** LIMITING STEPS!!! **')
            self.max_batch = len(self.trainloader_no_up)
        else:
            self.max_batch = None

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        # The only thing different from plain resnet is that init_feat_only is set to true now
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        # TODO: LOADING WARM HERE
        # self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
        #                      weights_init=self.args.weights_init, num_layers=self.args.num_layers,
        #                      init_feat_only=False, T=self.args.T, alpha=self.args.alpha)
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init.replace('.pth', '_hall_warm.pth'), num_layers=self.args.num_layers,
                             init_feat_only=False, T=self.args.T, alpha=self.args.alpha)

        self.set_optimizers()

    def set_optimizers(self):
        self.logger.info('** SETTING OPTIMIZERS!!! **')
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
                                       'lr': self.args.lr_classifier * 0.1,
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
                                   'lr': self.args.lr_memory,
                                   'momentum': self.args.momentum_memory,
                                   'weight_decay': self.args.weight_decay_memory}])
        self.sch_mem = optim.lr_scheduler.StepLR(self.opt_mem, step_size=self.args.step_size,
                                                 gamma=self.args.gamma)

    def reset_trainloader(self):

        self.logger.info('\nReseting training loader and sampler with pseudo labels.')

        cls_idx = class_indices[self.args.class_indices]

        self.logger.info('\nTRAINLOADER_UP_GT....')
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
                                              pseudo_labels_hard=None,
                                              pseudo_labels_soft=self.pseudo_labels_soft,
                                              GTPS_mode='GT',
                                              blur=True)

        self.logger.info('\nTRAINLOADER_UP_PS....')
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
                                              pseudo_labels_hard=self.pseudo_labels_hard,
                                              pseudo_labels_soft=self.pseudo_labels_soft,
                                              GTPS_mode='PS',
                                              blur=True)

        self.logger.info('\nTRAINLOADER_NO_UP_GT....')
        self.trainloader_no_up_gt = load_dataset(name=self.args.dataset_name,
                                                 class_indices=cls_idx,
                                                 dset='train',
                                                 transform='train_strong',
                                                 split=None,
                                                 rootdir=self.args.dataset_root,
                                                 batch_size=int(self.args.batch_size / 2),
                                                 shuffle=True,  # Here
                                                 num_workers=self.args.num_workers,
                                                 cas_sampler=False,  # Here
                                                 conf_preds=self.conf_preds,
                                                 pseudo_labels_hard=None,
                                                 pseudo_labels_soft=self.pseudo_labels_soft,
                                                 GTPS_mode='GT',
                                                 blur=True)

        self.logger.info('\nTRAINLOADER_NO_UP_PS....')
        self.trainloader_no_up_ps = load_dataset(name=self.args.dataset_name,
                                                 class_indices=cls_idx,
                                                 dset='train',
                                                 transform='train_strong',
                                                 split=None,
                                                 rootdir=self.args.dataset_root,
                                                 batch_size=int(self.args.batch_size / 2),
                                                 shuffle=True,  # Here
                                                 num_workers=self.args.num_workers,
                                                 cas_sampler=False,  # Here
                                                 conf_preds=self.conf_preds,
                                                 pseudo_labels_hard=self.pseudo_labels_hard,
                                                 pseudo_labels_soft=self.pseudo_labels_soft,
                                                 GTPS_mode='PS',
                                                 blur=True)

        self.logger.info('\nTRAINLOADER_NO_UP_GTPS....')
        self.trainloader_no_up_gtps = load_dataset(name=self.args.dataset_name,
                                                   class_indices=cls_idx,
                                                   dset='train',
                                                   transform='eval',
                                                   split=None,
                                                   rootdir=self.args.dataset_root,
                                                   batch_size=int(self.args.batch_size / 2),
                                                   shuffle=True,  # Here
                                                   num_workers=self.args.num_workers,
                                                   cas_sampler=False,  # Here
                                                   conf_preds=self.conf_preds,
                                                   pseudo_labels_hard=self.pseudo_labels_hard,
                                                   pseudo_labels_soft=self.pseudo_labels_soft,
                                                   GTPS_mode='both',
                                                   blur=False)

    def train_memory_epoch(self, epoch, soft=False):

        self.net.feature.train()
        self.net.fc_hallucinator.train()
        self.net.fc_selector.train()
        self.net.cosnorm_classifier.train()
        self.net.criterion_ctr.train()

        if epoch % self.args.no_up_freq == 0:
            loader_gt = self.trainloader_no_up_gt
            loader_ps = self.trainloader_no_up_ps
            up = False
        else:
            loader_gt = self.trainloader_up_gt
            loader_ps = self.trainloader_up_ps
            up = True

        iter_gt = iter(loader_gt)
        iter_ps = iter(loader_ps)

        # if self.max_batch is not None and (self.max_batch) * 2 < len(self.trainloader_up_ps):
        #     N = self.max_batch * 2
        # else:
        #     N = len(self.trainloader_up_ps)

        if up and self.max_batch is not None:
            N = self.max_batch * 2
        else:
            N = len(loader_ps)

        for batch_idx in range(N):

            # log basic adda train info
            info_str = '[Memory training (Stage 2)] '
            info_str += '[Soft] ' if soft else '[Hard] '
            info_str += '[up] ' if up else '[no_up] '
            info_str += 'Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                              N, 100 * batch_idx / N)

            try:
                data_gt, labels_gt, soft_target_gt, _, _ = next(iter_gt)
            except StopIteration:
                iter_gt = iter(loader_gt)
                data_gt, labels_gt, soft_target_gt, _, _ = next(iter_gt)

            data_ps, labels_ps, soft_target_ps, _, _ = next(iter_ps)

            data = torch.cat((data_gt, data_ps), dim=0)
            labels = torch.cat((labels_gt, labels_ps), dim=0)
            soft_target = torch.cat((soft_target_gt, soft_target_ps), dim=0)

            ########################
            # Setup data variables #
            ########################
            # assign devices
            data, labels, soft_target = data.cuda(), labels.cuda(), soft_target.cuda()
            data.requires_grad = False
            labels.requires_grad = False
            soft_target.requires_grad = False

            ####################
            # Forward and loss #
            ####################

            feats, logits, _ = self.memory_forward(data)

            preds = logits.argmax(dim=1)

            # calculate loss
            if soft:
                xent_loss = self.net.criterion_cls_soft(logits, labels, soft_target)
            else:
                xent_loss = self.net.criterion_cls_hard(logits, labels)
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

        best_semi_iter = 0
        best_epoch = 0
        best_acc = 0.

        self.logger.info('\nValidating initial model and reset pseudo labels.\n')
        _ = self.evaluate(self.valloader, hall=True, soft_reset=False, hard_reset=False)
        _ = self.evaluate(self.trainloader_no_up, hall=True, soft_reset=True, hard_reset=True)
        # _ = self.evaluate(self.valloader, hall=False, soft_reset=False, hard_reset=False)
        # _ = self.evaluate(self.trainloader_no_up, hall=False, soft_reset=True, hard_reset=True)

        for semi_i in range(self.args.semi_iters):

            # Setting new train loader with CAS
            self.reset_trainloader()

            # Generate centroids!!
            self.logger.info('\nCalculating initial centroids for all stage 2 classes.')
            initial_centroids = self.centroids_cal(self.trainloader_no_up_gtps).clone().detach()

            # Intitialize centroids using named parameter to avoid possible bugs
            with torch.no_grad():
                for name, param in self.net.criterion_ctr.named_parameters():
                    if name == 'centroids':
                        self.logger.info('\nPopulating initial centroids.\n')
                        param.copy_(initial_centroids)

            for epoch in range(self.args.oltr_epochs):

                self.train_memory_epoch(epoch, soft=(semi_i != 0))

                # Validation
                self.logger.info('\nValidation, semi-iteration {}.'.format(semi_i))
                val_acc_mac = self.evaluate(self.valloader)
                if val_acc_mac > best_acc:
                    self.logger.info('\nUpdating Best Model Weights!!')
                    self.net.update_best()
                    best_acc = val_acc_mac
                    best_epoch = epoch
                    best_semi_iter = semi_i
                self.logger.info('\nCurrrent Best Acc is {:.3f} at epoch {} semi-iter {}...'
                                 .format(best_acc * 100, best_epoch, best_semi_iter))

            # Revert to best weights
            self.net.load_state_dict(copy.deepcopy(self.net.best_weights))

            # Reset pseudo labels
            _ = self.evaluate(self.trainloader_no_up, hall=False, soft_reset=True, hard_reset=True)

            self.set_optimizers()

            self.logger.info('\nBest Model Appears at Epoch {} Semi-iteration {} with Acc {:.3f}...'
                             .format(best_epoch, best_semi_iter, best_acc * 100))

            self.save_model()

    def evaluate_epoch(self, loader, hall=False, soft_reset=False, hard_reset=False):

        self.net.eval()

        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()

        total_logits = []
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

            # for data, labels in tqdm(loader, total=len(loader)):

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
                    # reachability_logits = (18 / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                    # reachability_logits = (13 / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])

                    # if soft_reset or hard_reset:
                    #     reachability_logits = ((self.args.reachability_scale / values_nn[:, 0])
                    #                            .unsqueeze(1).expand(-1, logits.shape[1]))
                    # else:
                    #     reachability_logits = ((self.args.reachability_scale_eval / values_nn[:, 0])
                    #                            .unsqueeze(1).expand(-1, logits.shape[1]))
                    # logits = reachability_logits * logits

                # compute correct
                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                total_logits.append(logits.detach().cpu().numpy())
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

        total_max_probs = np.concatenate(total_max_probs, axis=0)
        conf_preds = np.zeros(len(total_max_probs))
        conf_preds[total_max_probs > self.args.theta] = 1

        if soft_reset:
            self.logger.info("** Reseting soft pseudo labels **\n")
            if self.pseudo_labels_soft is not None:
                self.pseudo_labels_soft[conf_preds == 1] = np.concatenate(total_logits, axis=0)[conf_preds == 1]
            else:
                self.pseudo_labels_soft = np.concatenate(total_logits, axis=0)
        if hard_reset:
            self.logger.info("** Reseting hard pseudo labels **\n")
            self.pseudo_labels_hard[conf_preds == 1] = np.concatenate(total_preds, axis=0)[conf_preds == 1]


        return eval_info, class_acc.mean()

    def evaluate(self, loader, hall=False, soft_reset=False, hard_reset=False):
        eval_info, eval_acc_mac = self.evaluate_epoch(loader, hall=hall, soft_reset=soft_reset, hard_reset=hard_reset)
        self.logger.info(eval_info)
        return eval_acc_mac

    def centroids_cal(self, loader):

        self.net.eval()

        centroids = torch.zeros(len(class_indices[self.args.class_indices]),
                                self.net.feature_dim).cuda()

        with torch.set_grad_enabled(False):

            for batch in tqdm(loader, total=len(loader)):

                if loader == self.trainloader_no_up_gtps:
                    data, labels, _, _, _ = batch
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
                # reachability_logits = ((self.args.reachability_scale_eval / values_nn[:, 0])
                #                        .unsqueeze(1).expand(-1, logits.shape[1]))
                # logits = reachability_logits * logits

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

    def deploy_ood_epoch(self, loader):

        self.net.eval()

        total_preds = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for data, file_id in tqdm(loader, total=len(loader)):

                # setup data
                data = data.cuda()
                data.requires_grad = False

                # forward
                _, logits, values_nn = self.memory_forward(data)

                # scale logits with reachability
                # reachability_logits = ((self.args.reachability_scale_eval / values_nn[:, 0])
                #                        .unsqueeze(1).expand(-1, logits.shape[1]))
                # reachability_logits = ((5 / values_nn[:, 0])
                #                        .unsqueeze(1).expand(-1, logits.shape[1]))
                # logits = reachability_logits * logits

                # compute correct
                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                # Set unconfident prediction to 1
                preds[max_probs < self.args.theta] = 1
                preds[max_probs >= self.args.theta] = 0

                total_preds.append(preds.detach().cpu().numpy())

        total_preds = np.concatenate(total_preds, axis=0)
        unconf_unknown_percent = total_preds.sum() / len(total_preds)

        eval_info = 'Unconf Unknown Percentage: {:3f}\n'.format(unconf_unknown_percent * 100)

        return eval_info

    def deploy_ood(self, loader):
        eval_info = self.deploy_ood_epoch(loader)
        self.logger.info(eval_info)

