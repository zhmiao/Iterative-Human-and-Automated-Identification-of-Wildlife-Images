import os
import numpy as np
import copy
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm, Algorithm, acc, WarmupScheduler, ood_metric
# from .plain_memory_stage_2_conf_pseu import PlainMemoryStage2_ConfPseu
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model


def load_data(args, conf_preds, pseudo_labels):

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
                                     pseudo_labels_hard=pseudo_labels,
                                     pseudo_labels_soft=None,
                                     GTPS_mode='both')

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

    return trainloader_no_up,  valloader, valloaderunknown, deployloader


@register_algorithm('SemiStage2OLTR')
class OLTR(Algorithm):

    """
    Overall training function.
    """

    name = 'SemiStage2OLTR'
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

        self.conf_preds = list(np.fromfile('./weights/EnergyStage1/101920_MOZ_S1_1_conf_preds.npy').astype(int))
        self.pseudo_labels_hard = np.fromfile('./weights/EnergyStage1/101920_MOZ_S1_1_init_pseudo_hard.npy',
                                              dtype=np.int)

        self.pseudo_labels_soft = None

        #######################################
        # Setup data for training and testing #
        #######################################
        (self.trainloader_eval, self.valloader, self.valloaderunknown,
         self.deployloader) = load_data(args, self.conf_preds, self.pseudo_labels_hard)

        self.train_class_counts = self.trainloader_eval.dataset.class_counts
        self.train_annotation_counts = self.trainloader_eval.dataset.class_counts_ann

        self.trainloader_up_gt = None
        self.trainloader_up_ps = None
        self.trainloader_no_up_gt = None
        self.trainloader_no_up_ps = None

        if args.limit_steps:
            self.logger.info('** LIMITING STEPS!!! **')
            self.max_batch = len(self.trainloader_eval)
        else:
            self.max_batch = None

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        # The only thing different from plain resnet is that init_feat_only is set to true now
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))

        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init, num_layers=self.args.num_layers,
                             init_feat_only=True, T=self.args.T, alpha=self.args.alpha)

        _ = self.evaluate(self.valloader, hall=True)

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

    def pseudo_label_reset(self, loader, hall=False, hard=False, soft=False):
        self.net.eval()
        total_preds, total_labels, total_logits, conf_preds = self.evaluate_forward(loader, hall=hall, 
                                                                                    ood=False, out_conf=True)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        total_logits = np.concatenate(total_logits, axis=0)

        if hard:
            self.logger.info("** Reseting hard pseudo labels **\n")
            self.pseudo_labels_hard[conf_preds == 1] = total_preds[conf_preds == 1]
        
        if soft:
            self.logger.info("** Reseting soft pseudo labels **\n")
            if self.pseudo_labels_soft is not None:
                self.pseudo_labels_soft[conf_preds == 1] = total_logits[conf_preds == 1]
            else:
                self.pseudo_labels_soft = total_logits

    def train(self):

        best_semi_iter = 0
        best_epoch = 0
        best_acc_mac = 0.
        best_acc_mic = 0.

        self.pseudo_label_reset(self.trainloader_eval, hall=True, soft=True, hard=True)

        for semi_i in range(self.args.semi_iters):

            # Setting new train loader with CAS
            self.reset_trainloader()

            # Generate centroids!!
            self.logger.info('\nCalculating initial centroids for all stage 2 classes.')
            initial_centroids = self.centroids_cal(self.trainloader_eval).clone().detach()

            # Intitialize centroids using named parameter to avoid possible bugs
            with torch.no_grad():
                for name, param in self.net.criterion_ctr.named_parameters():
                    if name == 'centroids':
                        self.logger.info('\nPopulating initial centroids.\n')
                        param.copy_(initial_centroids)

            for epoch in range(self.args.oltr_epochs):

                self.train_epoch(epoch, soft=(semi_i != 0))

                # Validation
                self.logger.info('\nValidation, semi-iteration {}.'.format(semi_i))
                val_acc_mac, val_acc_mic = self.evaluate(self.valloader)
                if val_acc_mac > best_acc_mac:
                    self.logger.info('\nUpdating Best Model Weights!!')
                    self.net.update_best()
                    best_acc_mac = val_acc_mac
                    best_acc_mic = val_acc_mic
                    best_epoch = epoch
                    best_semi_iter = semi_i
                self.logger.info('\nCurrrent Best Mac Acc is {:.3f} (Mic: {:.3f}) at epoch {} semi-iter {}...'
                                 .format(best_acc_mac * 100, best_acc_mic * 100, best_epoch, best_semi_iter))

            # Revert to best weights
            self.net.load_state_dict(copy.deepcopy(self.net.best_weights))

            # Reset pseudo labels
            self.pseudo_label_reset(self.trainloader_eval, hall=False, soft=True, hard=True)

            self.set_optimizers()

            self.logger.info('\nBest Model Appears at Epoch {} Semi-iteration {} with Mac Acc {:.3f} (Mic {:.3f})...'
                             .format(best_epoch, best_semi_iter, best_acc_mac * 100, best_acc_mic * 100))

            self.save_model()

    def evaluate(self, loader, hall=False, ood=False):
        if ood:
            eval_info, f1, _ = self.ood_evaluate_epoch(loader, self.valloaderunknown, hall=hall)
            self.logger.info(eval_info)
            return f1
        else:
            eval_info, eval_acc_mac, eval_acc_mic = self.evaluate_epoch(loader, hall=hall)
            self.logger.info(eval_info)
            return eval_acc_mac, eval_acc_mic

    def deploy(self, loader):
        eval_info, f1, conf_preds = self.deploy_epoch(loader)
        self.logger.info(eval_info)
        conf_preds_path = self.weights_path.replace('.pth', '_conf_preds.npy')
        self.logger.info('Saving confident predictions to {}'.format(conf_preds_path))
        conf_preds.tofile(conf_preds_path)
        return f1

    def train_epoch(self, epoch, soft=False):

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
                data_gt, labels_gt, soft_target_gt = next(iter_gt)
            except StopIteration:
                iter_gt = iter(loader_gt)
                data_gt, labels_gt, soft_target_gt = next(iter_gt)

            data_ps, labels_ps, soft_target_ps = next(iter_ps)

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

    def evaluate_epoch(self, loader, hall=False):
        self.net.eval()
        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()
        total_preds, total_labels, _ = self.evaluate_forward(loader, hall=hall, 
                                                             ood=False, out_conf=False)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        eval_info, mac_acc, mic_acc = self.evaluate_metric(total_preds, total_labels, 
                                                           eval_class_counts, ood=False)
        return eval_info, mac_acc, mic_acc

    def ood_evaluate_epoch(self, loader_in, loader_out, hall=False):
        self.net.eval()
        # Get unique classes in the loader and corresponding counts
        loader_uni_class_in, eval_class_counts_in = loader_in.dataset.class_counts_cal()
        loader_uni_class_out, eval_class_counts_out = loader_out.dataset.class_counts_cal()

        self.logger.info("Forward through in test loader\n")
        total_preds_in, total_labels_in, _ = self.evaluate_forward(loader_in, hall=hall,
                                                                   ood=True, out_conf=False)
        total_preds_in = np.concatenate(total_preds_in, axis=0)
        total_labels_in = np.concatenate(total_labels_in, axis=0)

        self.logger.info("Forward through out test loader\n")
        total_preds_out, total_labels_out, _ = self.evaluate_forward(loader_out, hall=hall,
                                                                     ood=True, out_conf=False)
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
        total_preds, total_labels, _ = self.evaluate_forward(loader, hall=False,
                                                             ood=True, out_conf=False)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        eval_info, f1, conf_preds = self.evaluate_metric(total_preds, total_labels, 
                                                         eval_class_counts, ood=True)
        return eval_info, f1, conf_preds

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

    def evaluate_forward(self, loader, hall=False, ood=False, out_conf=False):

        if hall: 
            self.logger.info("\n** Using hallucinator for evaluation **\n")
        
        total_preds = []
        total_labels = []
        total_logits = []
        total_probs = []

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
                    # forward
                    feats, logits, values_nn = self.memory_forward(data)

                    # feats_expand = meta_feats.clone().unsqueeze(1).expand(-1, len(self.meta_centroids), -1)
                    #
                    # centroids_expand = self.meta_centroids.clone().unsqueeze(0).expand(len(data), -1, -1)
                    #
                    # # computing reachability
                    # dist_to_centroids = torch.norm(feats_expand - centroids_expand, 2, 2)
                    # # Sort distances
                    # values_nn, labels_nn = torch.sort(dist_to_centroids, 1)
                    #
                    # min_dists = values_nn[:, 0]

                    # # expand to logits dimension and scale the smallest distance
                    # reachability = (20 / values_nn[:, 0]).unsqueeze(1).expand(-1,
                    # # scale logits with reachability
                    # logits = reachability * logits

                    # _, logits, values_nn = self.memory_forward(data)
                    # scale logits with reachability
                    # reachability_logits = (40 / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                    # reachability_logits = (18 / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                    # reachability_logits = (13 / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])


                    # if not soft_reset or not hard_reset:
                    #     logits /= torch.norm(logits, 2, 1, keepdim=True).clone()
                    #     reachability_logits = ((self.args.reachability_scale_eval / values_nn[:, 0])
                    #                            .unsqueeze(1).expand(-1, logits.shape[1]))
                    #     logits = reachability_logits * logits

                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                if ood:
                    # Set unconfident prediction to -1
                    preds[max_probs < self.args.theta] = -1

                total_preds.append(preds.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())
                total_logits.append(logits.detach().cpu().numpy())
                total_probs.append(max_probs.detach().cpu().numpy())

        if out_conf:
            total_probs = np.concatenate(total_probs, axis=0)
            conf_preds = np.zeros(len(total_probs))
            conf_preds[total_probs >= self.args.theta] = 1
            return total_preds, total_labels, total_logits, conf_preds
        else:
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

    def centroids_cal(self, loader):

        self.net.eval()

        centroids = torch.zeros(len(class_indices[self.args.class_indices]),
                                self.net.feature_dim).cuda()

        with torch.set_grad_enabled(False):

            for batch in tqdm(loader, total=len(loader)):

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

    def save_model(self):
        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path))
        self.net.save(self.weights_path)
