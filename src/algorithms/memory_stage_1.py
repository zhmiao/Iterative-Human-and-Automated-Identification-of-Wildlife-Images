import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm, Algorithm, stage_1_metric
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.data.class_aware_sampler import ClassAwareSampler
from src.models.utils import get_model

def load_data(args):

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
                                     sampler=None)

    # Use replace S1 to S2 for evaluation
    testloader = load_dataset(name=args.dataset_name.replace('S1', 'S2'),
                              class_indices=cls_idx,
                              dset='test',
                              transform='eval',
                              split=None,
                              rootdir=args.dataset_root,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              sampler=None)

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
                             sampler=None)

    return trainloader_no_up, testloader, valloader

@register_algorithm('MemoryStage1')
class MemoryStage1(Algorithm):

    """
    Overall training function.
    """

    name = 'MemoryStage1'
    net = None
    opt_net = None
    scheduler = None
    centroids = None

    def __init__(self, args):
        super(MemoryStage1, self).__init__(args=args)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader, self.testloader, self.valloader = load_data(args)
        _, self.train_class_counts = self.trainloader.dataset.class_counts_cal()

        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader_no_up, \
        self.testloader, self.valloader = load_data(args)
        self.train_unique_labels, self.train_class_counts = self.trainloader_no_up.dataset.class_counts_cal()

        self.trainloader_up = None
        self.reset_trainloader()

        if args.limit_steps:
            self.logger.info('** LIMITING STEPS!!! **')
            self.max_batch = len(self.trainloader_no_up)
        else:
            self.max_batch = None

    def reset_trainloader(self):
        self.logger.info('\nReseting training loader and sampler with pseudo labels.')
        cls_idx = class_indices[self.args.class_indices]
        sampler = ClassAwareSampler(labels=self.trainloader_no_up.dataset.labels,
                                    num_samples_cls=self.args.num_samples_cls)
        self.trainloader_up = load_dataset(name=self.args.dataset_name,
                                           class_indices=cls_idx,
                                           dset='train',
                                           transform=self.args.train_transform,
                                           split=None,
                                           rootdir=self.args.dataset_root,
                                           batch_size=self.args.batch_size,
                                           shuffle=False,
                                           num_workers=self.args.num_workers,
                                           sampler=sampler)

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
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

        # N = len(self.trainloader)

        if self.max_batch is not None:
            N = self.max_batch
        else:
            N = len(self.trainloader_up)

        for batch_idx, (data, labels) in enumerate(self.trainloader_up):

            if batch_idx > N:
                break

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

    def train(self):

        best_acc = 0.

        for epoch in range(self.num_epochs):

            # Training
            self.train_epoch(epoch)

            # Validation
            self.logger.info('\nValidation.')
            val_acc_mac = self.evaluate(self.valloader)
            if val_acc_mac > best_acc:
                self.net.update_best()

        self.save_model()

    def test_epoch(self, loader):

        self.net.eval()

        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()

        total_preds = []
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

                # Reachability
                # expand dimension
                feats_expand = feats.clone().unsqueeze(1).expand(-1, len(self.centroids), -1)
                centroids_expand = self.centroids.clone().unsqueeze(0).expand(len(data), -1, -1)
                # computing reachability
                dist_to_centroids = torch.norm(feats_expand - centroids_expand, 2, 2)
                # Sort distances
                values_nn, labels_nn = torch.sort(dist_to_centroids, 1)
                # expand to logits dimension and scale the smallest distance
                reachability = (self.args.reach_scale / values_nn[:, 0]).unsqueeze(1).expand(-1, logits.shape[1])
                # scale logits with reachability
                logits = reachability * logits

                # Prediction
                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                # Set unconfident prediction to -1
                preds[max_probs < self.args.theta] = -1

                total_preds.append(preds.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())

        f1,\
        class_acc_confident, class_percent_confident, false_pos_percent, \
        class_percent_wrong_unconfident, \
        percent_unknown, conf_preds = stage_1_metric(np.concatenate(total_preds, axis=0),
                                                     np.concatenate(total_labels, axis=0),
                                                     loader_uni_class,
                                                     eval_class_counts)

        eval_info = '{} Per-class evaluation results: \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

        for i in range(len(class_acc_confident)):
            eval_info += 'Class {} (train counts {}):'.format(i, self.train_class_counts[loader_uni_class][i])
            eval_info += 'Confident percentage: {:.2f};'.format(class_percent_confident[i] * 100)
            eval_info += 'Unconfident wrong %: {:.2f};'.format(class_percent_confident[i] * 100)
            eval_info += 'Accuracy: {:.3f} \n'.format(class_acc_confident[i] * 100)

        eval_info += 'Overall F1: {:.3f} \n'.format(f1)
        eval_info += 'False positive percentage: {:.3f} \n'.format(false_pos_percent * 100)
        eval_info += 'Selected unknown percentage: {:.3f} \n'.format(percent_unknown * 100)

        return eval_info, f1, conf_preds, np.concatenate(total_preds, axis=0)

    def validate_epoch(self, loader):

        self.net.eval()

        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()
        class_correct = np.array([0. for _ in range(len(eval_class_counts))])

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
                preds = logits.argmax(dim=1)
                for i in range(len(preds)):
                    pred = preds[i]
                    label = labels[i]
                    if pred == label:
                        class_correct[label] += 1

        # Record per class accuracies
        class_acc = class_correct[loader_uni_class] / eval_class_counts[loader_uni_class]
        overall_acc = class_correct.sum() / eval_class_counts.sum()
        eval_info = '{} Per-class evaluation results: \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        for i in range(len(class_acc)):
            eval_info += 'Class {} (train counts {}): {:.3f} \n'.format(i, self.train_class_counts[loader_uni_class][i],
                                                                        class_acc[i] * 100)

        # Record missing classes in evaluation sets if exist
        missing_classes = list(set(loader.dataset.class_indices.values()) - set(loader_uni_class))
        eval_info += 'Missing classes in evaluation set: '
        for c in missing_classes:
            eval_info += 'Class {} (train counts {})'.format(c, self.train_class_counts[c])

        return eval_info, class_acc.mean(), overall_acc

    def centroids_cal(self, loader):

        self.net.eval()

        centroids = torch.zeros(len(class_indices[self.args.class_indices]),
                                self.net.feature_dim).cuda()

        with torch.set_grad_enabled(False):
            for data, labels in tqdm(loader, total=len(loader)):
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

        if loader == self.testloader:

            # Calculate training data centroids first
            centroids_path = self.weights_path.replace('.pth', '_centroids.npy')
            if os.path.exists(centroids_path):
                self.logger.info('Loading centroids from {}.\n'.format(centroids_path))
                cent_np = np.fromfile(centroids_path, dtype=np.float32).reshape(-1, self.net.feature_dim)
                self.centroids = torch.from_numpy(cent_np).cuda()
            else:
                self.logger.info('Calculating training data centroids.\n')
                self.centroids = self.centroids_cal(self.trainloader)
                self.centroids.clone().detach().cpu().numpy().tofile(centroids_path)
                self.logger.info('Centroids saved to {}.\n'.format(centroids_path))

            # Evaluate
            eval_info, f1, conf_preds, init_pseudo = self.test_epoch(loader)
            self.logger.info(eval_info)

            conf_preds_path = self.weights_path.replace('.pth', '_conf_preds.npy')
            self.logger.info('Saving confident predictions to {}'.format(conf_preds_path))
            conf_preds.tofile(conf_preds_path)

            init_pseudo_path = self.weights_path.replace('.pth', '_init_pseudo.npy')
            self.logger.info('Saving initial pseudolabels to {}'.format(init_pseudo_path))
            init_pseudo.tofile(init_pseudo_path)

            return f1

        else:

            eval_info, eval_acc_mac, eval_acc_mic = self.validate_epoch(loader)
            self.logger.info(eval_info)
            self.logger.info('Macro Acc: {:.3f}; Micro Acc: {:.3f}\n'.format(eval_acc_mac * 100, eval_acc_mic * 100))

            return eval_acc_mac

    def save_model(self):
        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path))
        self.net.save(self.weights_path)






