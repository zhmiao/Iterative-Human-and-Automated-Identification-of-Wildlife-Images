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
from src.algorithms.stage_2_pslabel_oltr import OLTR


@register_algorithm('SemiStage2OLTR_Energy')
class OLTR_Energy(OLTR):

    """
    Overall training function.
    """

    name = 'SemiStage2OLTR_Energy'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(OLTR_Energy, self).__init__(args=args)

    def set_train(self):

        self.trainloaderunknown = load_dataset(name='MOZ_UNKNOWN',
                                               class_indices=class_indices[self.args.class_indices],
                                               dset='train',
                                               transform=self.args.train_transform,
                                               rootdir=self.args.dataset_root,
                                               batch_size=self.args.batch_size * 2,
                                               shuffle=True,
                                               num_workers=self.args.num_workers,
                                               cas_sampler=False)

        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        # The only thing different from plain resnet is that init_feat_only is set to true now
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))

        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.args.weights_init, num_layers=self.args.num_layers,
                             init_feat_only=False, T=self.args.T, alpha=self.args.alpha)

        self.set_optimizers()

        self.max_batch = len(self.trainloader_eval)

        # Reset pseudo labels
        self.pseudo_label_reset(self.trainloader_eval, hall=False, soft=False, hard=True)

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

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.logger.info('\nLoading from {}'.format(self.weights_path.replace('.pth', '_ft.pth')))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.weights_path.replace('.pth', '_ft.pth'),
                             num_layers=self.args.num_layers, init_feat_only=False)

        # self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
        #                      weights_init=self.args.weights_init,
        #                      num_layers=self.args.num_layers, init_feat_only=False)

    def set_optimizers(self):
        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos(step / total_steps * np.pi))

        def cosine_scheduler(opt, num_epochs, num_batchs, init_lr):
            return torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    num_epochs * num_batchs,
                    1,  # since lr_lambda computes multiplicative factor
                    1e-6 / init_lr))

        self.logger.info('** SETTING OPTIMIZERS!!! **')
        ######################
        # Optimization setup #
        ######################
        # Setup optimizer and optimizer scheduler
        self.opt_feats = optim.SGD([{'params': self.net.feature.parameters(),
                                     'lr': self.args.lr_feature,
                                     'momentum': self.args.momentum_feature,
                                     'weight_decay': self.args.weight_decay_feature}])
        self.sch_feats = cosine_scheduler(self.opt_feats, self.args.num_epochs, 
                                          len(self.trainloader_eval), self.args.lr_feature)

        self.opt_fc_hall = optim.SGD([{'params': self.net.fc_hallucinator.parameters(),
                                       'lr': self.args.lr_classifier * 0.1,
                                       'momentum': self.args.momentum_classifier,
                                       'weight_decay': self.args.weight_decay_classifier}])
        self.sch_fc_hall = cosine_scheduler(self.opt_fc_hall, self.args.num_epochs, 
                                          len(self.trainloader_eval), self.args.lr_feature)

        self.opt_fc_sel = optim.SGD([{'params': self.net.fc_selector.parameters(),
                                      'lr': self.args.lr_classifier,
                                      'momentum': self.args.momentum_classifier,
                                      'weight_decay': self.args.weight_decay_classifier}])
        self.sch_fc_sel = cosine_scheduler(self.opt_fc_sel, self.args.num_epochs, 
                                          len(self.trainloader_eval), self.args.lr_feature)

        self.opt_cos_clf = optim.SGD([{'params': self.net.cosnorm_classifier.parameters(),
                                       'lr': self.args.lr_classifier,
                                       'momentum': self.args.momentum_classifier,
                                       'weight_decay': self.args.weight_decay_classifier}])
        self.sch_cos_clf = cosine_scheduler(self.opt_cos_clf, self.args.num_epochs, 
                                          len(self.trainloader_eval), self.args.lr_feature)

        self.opt_mem = optim.SGD([{'params': self.net.criterion_ctr.parameters(),
                                   'lr': self.args.lr_memory,
                                   'momentum': self.args.momentum_memory,
                                   'weight_decay': self.args.weight_decay_memory}])
        self.sch_mem = cosine_scheduler(self.opt_mem, self.args.num_epochs, 
                                          len(self.trainloader_eval), self.args.lr_feature)

    def reset_trainloader(self):

        self.logger.info('\nReseting training loader and sampler with pseudo labels.')

        cls_idx = class_indices[self.args.class_indices]

        self.logger.info('\nTRAINLOADER_EVAL....')
        self.trainloader_eval = load_dataset(name=self.args.dataset_name,
                                             class_indices=cls_idx,
                                             dset='train',
                                             transform='eval',
                                             rootdir=self.args.dataset_root,
                                             batch_size=self.args.batch_size,
                                             shuffle=False,
                                             num_workers=self.args.num_workers,
                                             cas_sampler=False,
                                             conf_preds=self.conf_preds,
                                             pseudo_labels_hard=self.pseudo_labels_hard,
                                             pseudo_labels_soft=None,
                                             GTPS_mode='both')

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
                                              pseudo_labels_soft=None,
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
                                              pseudo_labels_soft=None,
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

    def energy_ft(self):
        best_f1 = 0

        for epoch in range(self.args.num_epochs):
            # Training
            self.energy_ft_epoch(epoch)
            # Validation
            self.logger.info('\nValidation.')
            f1 = self.evaluate(self.valloader, hall=False, ood=True)
            
            if f1 > best_f1:
                self.logger.info('\nUpdating Best Model Weights!!')
                self.net.update_best()
                best_f1 = f1

        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path.replace('.pth', '_ft.pth')))
        self.net.save(self.weights_path.replace('.pth', '_ft.pth'))

    def energy_ft_epoch(self, epoch):

        self.net.feature.train()
        self.net.fc_hallucinator.train()
        self.net.fc_selector.train()
        self.net.cosnorm_classifier.train()
        self.net.criterion_ctr.train()

        if epoch % self.args.no_up_freq == 0:
            loader_gt = self.trainloader_no_up_gt
            loader_ps = self.trainloader_no_up_ps
            up = False
            N = self.max_batch
        else:
            loader_gt = self.trainloader_up_gt
            loader_ps = self.trainloader_up_ps
            up = True
            N = self.max_batch * 2

        iter_gt = iter(loader_gt)
        iter_ps = iter(loader_ps)
        out_iter = iter(self.trainloaderunknown)


        for batch_idx in range(N):

            # log basic adda train info
            info_str = '[Energy Memory training (Stage 2)] '
            info_str += '[Hard] '
            info_str += '[up] ' if up else '[no_up] '
            info_str += 'Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                              N, 100 * batch_idx / N)
            
            # IN
            try:
                data_gt, labels_gt = next(iter_gt)
            except StopIteration:
                iter_gt = iter(loader_gt)
                data_gt, labels_gt = next(iter_gt)

            data_ps, labels_ps = next(iter_ps)

            data_in = torch.cat((data_gt, data_ps), dim=0)
            labels = torch.cat((labels_gt, labels_ps), dim=0)

            # OUT
            try:
                data_out, _ = next(out_iter)
            except StopIteration:
                out_iter = iter(self.trainloaderunknown)
                data_out, _ = next(out_iter)

            data = torch.cat((data_in, data_out), dim=0)

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

            # calculate oltr loss
            xent_loss = self.net.criterion_cls_hard(logits[:len(data_in)], labels)
            ctr_loss = self.net.criterion_ctr(feats[:len(data_in)].clone(), labels)

            oltr_loss = xent_loss + self.args.ctr_loss_weight * ctr_loss

            # calculate energy loss
            Ec_out = -torch.logsumexp(logits[len(data_in):], dim=1)
            Ec_in = -torch.logsumexp(logits[:len(data_in)], dim=1)
            m_out = -7.
            m_in = -18.
            eb_loss = torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean()
            
            
            loss = oltr_loss + 0.001 * eb_loss

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

            self.sch_feats.step()
            self.sch_fc_hall.step()
            self.sch_fc_sel.step()
            self.sch_cos_clf.step()
            self.sch_mem.step()

            ###########
            # Logging #
            ###########
            if batch_idx % self.log_interval == 0:
                # compute overall acc
                preds = logits[:len(data_in)].argmax(dim=1)
                acc = (preds == labels).float().mean()
                # log update info
                info_str += 'Acc: {:0.1f} Oltr: {:.3f}, EB: {:.3f}'.format(acc.item() * 100, oltr_loss.item(),
                                                                           eb_loss.item())
                self.logger.info(info_str)

    def evaluate(self, loader, hall=False, ood=False):
        if ood:
            # the = 6.77
            # T = 0.06
            # self.logger.info('\nCurrent T: {}'.format(T))
            # self.logger.info('\nCurrent the: {}'.format(the))
            eval_info, f1, _ = self.ood_evaluate_epoch(loader, self.valloaderunknown, hall=hall)
            self.logger.info(eval_info)
            
            return f1
        else:
            eval_info, eval_acc_mac, eval_acc_mic = self.evaluate_epoch(loader, hall=hall)
            self.logger.info(eval_info)
            return eval_acc_mac, eval_acc_mic

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

                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                if ood:
                    energy_score = -(self.args.energy_T * torch.logsumexp(logits / self.args.energy_T, dim=1))
                    # Set unconfident prediction to -1
                    preds[-energy_score <= self.args.energy_the] = -1

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

    def deploy_epoch(self, loader):
        self.net.eval()
        total_file_id = []
        total_preds = []
        total_max_probs = []
        total_energy = []
        total_probs = []

        with torch.set_grad_enabled(False):
            for data, file_id in tqdm(loader, total=len(loader)):

                # setup data
                data = data.cuda()
                data.requires_grad = False

                # forward
                _, logits, values_nn = self.memory_forward(data)

                # compute correct
                probs = F.softmax(logits, dim=1)
                max_probs, preds = probs.max(dim=1)

                energy_score = -(self.args.energy_T * torch.logsumexp(logits / self.args.energy_T, dim=1))

                total_preds.append(preds.detach().cpu().numpy())
                total_max_probs.append(max_probs.detach().cpu().numpy())
                total_file_id.append(file_id)
                total_energy.append(energy_score.detach().cpu().numpy())
                total_probs.append(probs.detach().cpu().numpy())

        total_file_id = np.concatenate(total_file_id, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)
        total_max_probs = np.concatenate(total_max_probs, axis=0)
        total_energy = np.concatenate(total_energy, axis=0)
        total_probs = np.concatenate(total_probs, axis=0)


        eval_info = '{} Picking Non-empty samples... \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

        total_preds[total_probs[:, 0] > 0.1] = 0

        total_file_id_em = total_file_id[total_preds == 0]

        total_file_id_ne = total_file_id[total_preds != 0]
        total_preds_ne = total_preds[total_preds != 0]
        total_max_probs_ne = total_max_probs[total_preds != 0]
        total_energy_ne = total_energy[total_preds != 0]

        eval_info += ('Total empty count is {} out of {} samples ({:3f}%) \n'
                      .format(len(total_file_id_em), len(total_preds), 100 * (len(total_file_id_em) / len(total_preds))))

        eval_info += '{} Picking confident samples... \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

        conf_preds = np.zeros(len(total_preds_ne))
        conf_preds[-total_energy_ne > self.args.energy_the] = 1

        total_file_id_conf = total_file_id_ne[conf_preds == 1]
        total_preds_conf = total_preds_ne[conf_preds == 1]

        total_file_id_unconf = total_file_id_ne[conf_preds == 0]
        total_preds_unconf = total_preds_ne[conf_preds == 0]

        eval_info += ('Total confident sample count is {} out of {} non-empty samples ({:3f}%) \n'
                      .format(len(total_preds_conf), len(total_preds_ne), 100 * (len(total_preds_conf) / len(total_preds_ne))))
        eval_info += ('Total unconfident sample count is {} out of {} non-empty samples ({:3f}%) \n'
                      .format(len(total_preds_unconf), len(total_preds_ne), 100 * (len(total_preds_unconf) / len(total_preds_ne))))

        eval_info += ('Total confident sample count is {} out of {} samples ({:3f}%) \n'
                      .format((len(total_preds_conf) + len(total_file_id_em)), len(total_preds),
                              100 * ((len(total_preds_conf) + len(total_file_id_em)) / len(total_preds))))

        return eval_info, (total_file_id_conf, total_preds_conf), (total_file_id_unconf, total_preds_unconf)
