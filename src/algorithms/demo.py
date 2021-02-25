import os
import numpy as np
import copy
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile

import torch
import torch.optim as optim
import torch.nn.functional as F

from .utils import register_algorithm, Algorithm, acc, WarmupScheduler, ood_metric
# from .plain_memory_stage_2_conf_pseu import PlainMemoryStage2_ConfPseu
from src.data.utils import load_dataset
from src.data.class_indices import class_indices
from src.models.utils import get_model
from src.algorithms.stage_2_pslabel_oltr import OLTR


@register_algorithm('DEMO')
class DEMO(OLTR):

    """
    Inference Demo.
    """

    name = 'DEMO'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        self.args = args
        self.logger = self.args.logger
        self.weights_path = self.args.inference_weights

        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)

        # Training epochs and logging intervals
        self.log_interval = args.log_interval

        #######################################
        # Setup data for training and testing #
        #######################################
        self.cls_idx = class_indices[self.args.class_indices]
        self.deployloader = load_dataset(name=self.args.deploy_dataset_name,
                                         class_indices=self.cls_idx,
                                         dset=None,
                                         transform='eval',
                                         rootdir=self.args.dataset_root,
                                         batch_size=self.args.batch_size,
                                         shuffle=False,
                                         num_workers=self.args.num_workers,
                                         cas_sampler=False)

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.logger.info('\nLoading from {}'.format(self.weights_path))
        self.net = get_model(name=self.args.model_name, num_cls=len(class_indices[self.args.class_indices]),
                             weights_init=self.weights_path,
                             num_layers=self.args.num_layers, init_feat_only=False)

    def demo_inference(self, loader):
        eval_info, preds_conf, preds_unconf = self.deploy_epoch(loader)
        self.logger.info(eval_info)
        idx_cls = {item[1]: item[0] for item in self.cls_idx.items()}

        self.logger.info('Saving confident predictions to ./results/confident/...\n')
        os.makedirs('./results/confident', exist_ok=True)
        for file_id, pred in zip(*preds_conf):
            copyfile(os.path.join('./demo_data/data/', file_id),
                     os.path.join('./results/confident', '{}_'.format(idx_cls[pred]) + file_id))

        self.logger.info('Saving unconfident predictions to ./results/unconfident/...\n')
        os.makedirs('./results/unconfident', exist_ok=True)
        for file_id, pred in zip(*preds_unconf):
            copyfile(os.path.join('./demo_data/data/', file_id),
                     os.path.join('./results/unconfident', '{}_'.format(idx_cls[pred]) + file_id))

        self.logger.info('\nUnconfident predictions will be assigned to human annotation and confident predictions will be trusted.')

    def deploy_epoch(self, loader):

        self.net.eval()
        total_file_id = []
        total_labels = []
        total_preds = []
        total_max_probs = []
        total_energy = []
        total_probs = []
        total_feats = []

        with torch.set_grad_enabled(False):
            for data, file_id, labels in tqdm(loader, total=len(loader)):

                # setup data
                data = data.cuda()
                labels = labels.cuda()
                data.requires_grad = False
                labels.requires_grad = False

                # forward
                _, logits, values_nn, meta_feats = self.memory_forward(data)

                # compute correct
                probs = F.softmax(logits, dim=1)
                max_probs, preds = probs.max(dim=1)

                energy_score = -(self.args.energy_T * torch.logsumexp(logits / self.args.energy_T, dim=1))

                total_preds.append(preds.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())
                total_max_probs.append(max_probs.detach().cpu().numpy())
                total_file_id.append(file_id)
                total_energy.append(energy_score.detach().cpu().numpy())
                total_probs.append(probs.detach().cpu().numpy())
                total_feats.append(meta_feats.detach().cpu().numpy())

        total_file_id = np.concatenate(total_file_id, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        total_max_probs = np.concatenate(total_max_probs, axis=0)
        total_energy = np.concatenate(total_energy, axis=0)
        total_probs = np.concatenate(total_probs, axis=0)
        total_feats = np.concatenate(total_feats, axis=0)

        eval_info = '{} Picking confident samples... \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

        conf_preds = np.zeros(len(total_preds))
        conf_preds[-total_energy > self.args.energy_the] = 1

        total_file_id_conf = total_file_id[conf_preds == 1]
        total_preds_conf = total_preds[conf_preds == 1]
        total_labels_conf = total_labels[conf_preds == 1]

        total_file_id_unconf = total_file_id[conf_preds == 0]
        total_preds_unconf = total_preds[conf_preds == 0]

        eval_info += ('Total confident sample count is {} out of {} samples ({:.3f}%) \n'
                      .format(len(total_preds_conf), len(total_preds), 100 * (len(total_preds_conf) / len(total_preds))))
        eval_info += ('Total unconfident sample count is {} out of {} samples ({:.3f}%) \n'
                      .format(len(total_preds_unconf), len(total_preds), 100 * (len(total_preds_unconf) / len(total_preds))))
        eval_info += ('Confident prediction accuracy is : {:.3f}% \n'
                      .format(100 * (total_preds_conf == total_labels_conf).sum() / len(total_preds_conf)))

        return eval_info, (total_file_id_conf, total_preds_conf), (total_file_id_unconf, total_preds_unconf)
