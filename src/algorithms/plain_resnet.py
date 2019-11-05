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

    # def __init__(self, dataset_root, dataset_name='CCT_cis', model_name='PlainResNetClassifier'):


        self.num_epochs = args.num_epochs

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader = load_dataset(name=args.dataset_name, dset='train', rootdir=args.dataset_root,
                                        batch_size=64, shuffle=True, num_workers=1)

        self.testloader = load_dataset(name=args.dataset_name, dset='test', rootdir=args.dataset_root,
                                       batch_size=64, shuffle=True, num_workers=1)

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
        opt_net = optim.SGD(net_optim_params_list)

    def train_epoch(self):
        pass

    def test_epoch(self):
        pass

    def train(self):
        pass
        ##############
        # Train Adda #
        ##############
        # for epoch in range(num_epoch):
        #     err = train_epoch(train_src_data, train_tgt_data, net, opt_net, opt_dis, epoch)
        #     if err == -1:
        #         print("No suitable discriminator")
        #         break

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
