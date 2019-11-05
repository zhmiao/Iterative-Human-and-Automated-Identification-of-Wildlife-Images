import os
import yaml

import torch.optim as optim

from .utils import register_algorithm
from src.data.dataloader import load_dataset
from src.models.utils import get_model


@register_algorithm('PlainResNet')
class PlainResNet:

    """
    Overall training function.
    """

    name = 'PlainResNet'


    def __init__(self, args):

        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval
        self.logger = args.logger

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader = load_dataset(name=args.dataset_name, dset='train', rootdir=args.dataset_root,
                                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        self.testloader = load_dataset(name=args.dataset_name, dset='test', rootdir=args.dataset_root,
                                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        self.net = get_model(name=args.model_name, num_cls=args.num_cls,
                             weights_init=args.weights_init, num_layers=args.num_layers)
        # print network and arguments
        print(self.net)

        ######################
        # Optimization setup #
        ######################
        net_optim_params_list = [
            {'params': self.net.feature.parameters(),
             'lr': args.lr_feature,
             'momentum': args.momentum_feature,
             'weight_decay': args.weight_decay_feature},
            {'params': self.net.classifier.parameters(),
             'lr': args.lr_classifier,
             'momentum': args.momentum_classifier,
             'weight_decay': args.weight_decay_classifier}
        ]
        self.opt_net = optim.SGD(net_optim_params_list)
        # TODO: scheduler

    def train_epoch(self, epoch):

        self.net.train()

        N = len(self.trainloader)

        for batch_idx, (data, labels) in enumerate(self.trainloader):

            # log basic adda train info
            info_str = '[Train plain] Epoch: {} [{}/{} ({:.2f}%)] '.format(epoch, batch_idx,
                                                                           N, 100 * batch_idx / N)

            ########################
            # Setup data variables #
            ########################
            data, labels = data.cuda(), labels.cuda()

            data.require_grad = False
            labels.require_grad = False

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
                # compute discriminator acc
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean()
                # log discriminator update info
                info_str += 'Acc: {:0.1f} Xent: {:.3f}'.format(acc.item() * 100, loss.item())
                self.logger.info(info_str)

    def test_epoch(self):
        pass

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)

    def save_model(self):
        pass
        ##############
        # Save Model #
        ##############
        # os.makedirs(outdir, exist_ok=True)
        # outfile = join(outdir, 'adda_{:s}_net_{:s}_{:s}.pth'.format(
        #     model, src, tgt))
        # print('Saving to', outfile)
        # net.save(outfile)
