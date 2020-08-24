import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .utils import register_algorithm
from src.algorithms.plain_resnet import PlainResNet
from src.data.utils import load_dataset
from src.data.class_indices import class_indices


def load_data(args):

    """
    Dataloading function. This function can change alg by alg as well.
    """

    print('Using class indices: {} \n'.format(class_indices[args.class_indices]))

    trainloader = load_dataset(name=args.dataset_name,
                               class_indices=class_indices[args.class_indices],
                               dset='train',
                               transform='train',
                               split=args.train_split,
                               rootdir=args.dataset_root,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers)

    testloader = load_dataset(name=args.dataset_name,
                              class_indices=class_indices[args.class_indices],
                              dset='test',
                              transform='eval',
                              split=None,
                              rootdir=args.dataset_root,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)

    valloader = load_dataset(name=args.dataset_name,
                             class_indices=class_indices[args.class_indices],
                             dset='val',
                             transform='eval',
                             split=None,
                             rootdir=args.dataset_root,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    deployloader = load_dataset(name=args.deploy_dataset_name,
                                class_indices=class_indices[args.class_indices],
                                dset='deploy',
                                transform='eval',
                                split=None,
                                rootdir=args.dataset_root,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers)

    return trainloader, testloader, valloader, deployloader


@register_algorithm('EmptyPicker')
class EmptyPicker(PlainResNet):

    """
    Overall training function.
    """

    name = 'EmptyPicker'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):

        self.args = args
        self.logger = self.args.logger
        self.weights_path = './weights/{}/{}_{}.pth'.format(self.args.algorithm, self.args.conf_id, self.args.session)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader, self.testloader, self.valloader, self.deployloader = load_data(args)
        _, self.train_class_counts = self.trainloader.dataset.class_counts_cal()

    def deploy_epoch(self, loader):

        self.net.eval()

        total_preds = []
        total_paths = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for data, paths in tqdm(loader, total=len(loader)):

                # setup data
                data = data.cuda()
                data.requires_grad = False

                # forward
                feats = self.net.feature(data)
                logits = self.net.classifier(feats)

                # prediction
                max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                preds[max_probs < 0.8] = 0

                total_preds.append(preds.detach().cpu().numpy())
                total_paths.append(paths)

        total_preds = np.concatenate(total_preds, axis=0)
        total_paths = np.concatenate(total_paths, axis=0)

        return total_preds, total_paths

    def deploy(self, loader):
        total_preds, total_paths = self.deploy_epoch(loader)
        non_empty_paths = total_paths[total_preds == 1]
        nep_save_path = os.path.join(self.args.dataset_root, 'Mozambique/SplitLists/Mozambique_season_3_NEP.txt')
        self.logger.info('Non-empty file list saved to {}....\n'.format(nep_save_path))
        with open(nep_save_path, 'w') as txt:
            for p in non_empty_paths:
                txt.write(p + '\n')




