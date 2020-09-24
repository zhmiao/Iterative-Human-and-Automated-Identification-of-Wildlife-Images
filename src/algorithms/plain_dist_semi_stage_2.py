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

    return trainloader_no_up, testloader, valloader, deployloader


@register_algorithm('PlainDistSemiStage2')
class PlainDistSemiStage2(Algorithm):

    """
    Overall training function.
    """

    name = 'PlainDistSemiStage2'
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
        self.trainloader_no_up, self.testloader, self.valloader, self.deployloader = load_data(args, self.conf_preds)

        self.trainloader_no_up_gt = None
        self.trainloader_no_up_ps = None
        self.trainloader_no_up_gtps = None

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
                             weights_init=self.args.weights_init,
                             num_layers=self.args.num_layers, init_feat_only=True,
                             T=self.args.T, alpha=self.args.alpha)

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

    def reset_trainloader(self):

        self.logger.info('\nReseting training loader and sampler with pseudo labels.')

        cls_idx = class_indices[self.args.class_indices]

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

    def train_epoch(self, epoch, soft=False):

        self.net.train()

        loader_gt = self.trainloader_no_up_gt
        loader_ps = self.trainloader_no_up_ps
        up = False

        iter_gt = iter(loader_gt)
        iter_ps = iter(loader_ps)

        N = len(loader_ps)

        for batch_idx in range(N):

        # for batch_idx, (data, labels, _, _) in enumerate(self.trainloader):

            # log basic adda train info
            info_str = '[Train plain dist semi (Stage 2)] '
            info_str += '[Soft] ' if soft else '[Hard] '
            info_str += '[up] ' if up else '[no_up] '
            info_str += 'Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                              N, 100 * batch_idx / N)


            ########################
            # Setup data variables #
            ########################
            try:
                data_gt, labels_gt, soft_target_gt, _, _ = next(iter_gt)
            except StopIteration:
                iter_gt = iter(loader_gt)
                data_gt, labels_gt, soft_target_gt, _, _ = next(iter_gt)

            data_ps, labels_ps, soft_target_ps, _, _ = next(iter_ps)

            data = torch.cat((data_gt, data_ps), dim=0)
            labels = torch.cat((labels_gt, labels_ps), dim=0)
            soft_target = torch.cat((soft_target_gt, soft_target_ps), dim=0)
            # assign devices
            data, labels, soft_target = data.cuda(), labels.cuda(), soft_target.cuda()
            data.requires_grad = False
            labels.requires_grad = False
            soft_target.requires_grad = False

            # ########################
            # # Setup data variables #
            # ########################
            # data, labels = data.cuda(), labels.cuda()
            #
            # data.requires_grad = False
            # labels.requires_grad = False

            ####################
            # Forward and loss #
            ####################
            # forward
            feats = self.net.feature(data)
            logits = self.net.classifier(feats)
            # calculate loss
            if soft:
                loss = self.net.criterion_cls_soft(logits, labels, soft_target)
            else:
                loss = self.net.criterion_cls_hard(logits, labels)

            # # calculate loss
            # loss = self.net.criterion_cls(logits, labels)

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

        best_semi_iter = 0
        best_epoch = 0
        best_acc = 0.

        # self.logger.info('\nValidating initial model.\n')
        _ = self.evaluate(self.trainloader_no_up, soft_reset=True, hard_reset=False)
        # _ = self.evaluate(self.valloader, soft_reset=False, hard_reset=False)
        # _ = self.evaluate(self.trainloader_no_up, hall=True, soft_reset=True, hard_reset=True)
        # _ = self.evaluate(self.valloader, hall=False, soft_reset=False, hard_reset=False)
        # _ = self.evaluate(self.trainloader_no_up, hall=False, soft_reset=True, hard_reset=True)

        for semi_i in range(self.args.semi_iters):

            self.reset_trainloader()

            for epoch in range(self.args.num_epochs):

                self.train_epoch(epoch, soft=(semi_i != 0 and semi_i != 1))

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
            _ = self.evaluate(self.trainloader_no_up, soft_reset=True, hard_reset=True)

            self.set_optimizers()

            self.logger.info('\nBest Model Appears at Epoch {} Semi-iteration {} with Acc {:.3f}...'
                             .format(best_epoch, best_semi_iter, best_acc * 100))

            self.save_model()

    def evaluate_epoch(self, loader, soft_reset=False, hard_reset=False):

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

                # setup data
                data, labels = data.cuda(), labels.cuda()
                data.requires_grad = False
                labels.requires_grad = False

                # forward
                feats = self.net.feature(data)
                logits = self.net.classifier(feats)

                # Reachability
                # expand dimension
                feats_expand = feats.clone().unsqueeze(1).expand(-1, len(self.centroids), -1)
                centroids_expand = self.centroids.clone().unsqueeze(0).expand(len(data), -1, -1)
                # computing reachability
                dist_to_centroids = torch.norm(feats_expand - centroids_expand, 2, 2)
                # Sort distances
                values_nn, labels_nn = torch.sort(dist_to_centroids, 1)
                # expand to logits dimension and scale the smallest distance
                reachability = (self.args.reachability_scale_eval / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                # scale logits with reachability
                logits = reachability * logits

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

    def evaluate(self, loader, soft_reset=False, hard_reset=False):

        self.logger.info('Calculating training data centroids.\n')
        try:
            self.centroids = self.centroids_cal(self.trainloader_no_up_gtps)
        except:
            self.logger.info('Non-gtps.\n')
            self.centroids = self.centroids_cal(loader)

        eval_info, eval_acc_mac = self.evaluate_epoch(loader, soft_reset=soft_reset, hard_reset=hard_reset)
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
                elif loader == self.trainloader_no_up:
                    data, labels, _, _ = batch
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
                reachability_logits = ((self.args.reachability_scale_eval / values_nn[:, 0])
                                       .unsqueeze(1).expand(-1, logits.shape[1]))
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
