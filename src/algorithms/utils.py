import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


algorithms = {}
def register_algorithm(name):

    """
    Algorithm register
    """

    def decorator(cls):
        algorithms[name] = cls
        return cls
    return decorator


def get_algorithm(name, args):

    """
    Algorithm getter
    """

    alg = algorithms[name](args)
    return alg


class Algorithm:

    """
    Base Algorithm class for reference.
    """

    name = None

    def __init__(self, args):
        self.args = args
        self.logger = self.args.logger
        self.weights_path = './weights/{}/{}_{}.pth'.format(self.args.algorithm, self.args.conf_id, self.args.session)

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def train_epoch(self, epoch):
        pass

    def train(self):
        pass

    def evaluate_epoch(self, loader):
        pass

    def evaluate(self, loader):
        pass

    def deploy(self, loader):
        pass

    def save_model(self):
        pass


class WarmupScheduler:
    def __init__(self, optimizer, decay1, decay2, gamma, len_epoch, warmup_epochs=5, epi=1):
        self.optimizer = optimizer
        self.decay1 = decay1
        self.decay2 = decay2
        self.gamma = gamma
        self.len_epoch = len_epoch
        self.warmup_epochs = warmup_epochs
        self.epi = epi
        self.epoch = 1

        self.init_lr_list = []
        for param_group in self.optimizer.param_groups:
            self.init_lr_list.append(deepcopy(param_group['lr']))

    # def step(self, epoch, step):
    def step(self):

        """Sets the learning rate to the initial LR decayed by 10 every X epochs"""

        for i, param_group in enumerate(self.optimizer.param_groups):

            init_lr = self.init_lr_list[i]

            if self.epoch < (self.warmup_epochs * self.epi):
                lr = init_lr * self.epoch / self.warmup_epochs
            elif self.epoch >= (self.decay2 * self.epi):
                lr = init_lr * self.gamma * self.gamma
            elif self.epoch >= (self.decay1 * self.epi):
                lr = init_lr * self.gamma
            else:
                lr = init_lr

            param_group['lr'] = lr

        self.epoch += 1


def acc(preds, labels):

    _, label_counts = np.unique(labels, return_counts=True)

    class_correct = np.array([0. for _ in range(len(label_counts))])

    for p, l in zip(preds, labels):
        if p == l:
            class_correct[l] += 1

    class_acc = class_correct / label_counts

    mac_acc = class_acc.mean()
    mic_acc = class_correct.sum() / label_counts.sum()

    return class_acc, mac_acc, mic_acc


def f_measure(preds, labels):
    # f1 score for openset evaluation with close set accuracy
    true_pos = 0.
    false_pos = 0.
    false_neg = 0.

    for i in range(len(labels)):
        true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
        false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 else 0
        false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return 2 * ((precision * recall) / (precision + recall + 1e-12))


def confident_metrics(preds, labels, class_counts):
    # Confident metrics
    class_correct_confident = np.array([0. for _ in range(len(np.unique(labels)) - 1)])
    class_select_confident = np.array([0. for _ in range(len(np.unique(labels)) - 1)])
    preds_confident = preds[preds != -1]
    labels_confident = labels[preds != -1]

    false_pos_counts = 0.

    # compute correct predictions in confident preds
    for i in range(len(preds_confident)):
        pred = preds_confident[i]
        label = labels_confident[i]

        if label != -1:
            # Confident known accuracy
            if pred == label:
                class_correct_confident[label] += 1
            class_select_confident[label] += 1
        else:
            # Counts of unknown in confident set
            false_pos_counts += 1

    # Record per class accuracies for confident data
    class_acc_confident = class_correct_confident / class_select_confident
    class_percent_confident = class_select_confident / class_counts[1:]
    false_pos_percent = false_pos_counts / len(labels_confident)
    total_known = len(labels[labels != -1])

    return class_acc_confident, class_percent_confident, false_pos_percent, total_known


def unconfident_metrics(preds, labels):

    num_cls = len(np.unique(labels)) - 1

    class_unconf_wrong = np.array([0. for _ in range(num_cls)])
    class_wrong = np.array([1e-7 for _ in range(num_cls)])

    for i in range(len(preds)):

        pred = preds[i]
        label = labels[i]

        if pred != label and label != -1:
            class_wrong[label] += 1
            if pred == -1:
                class_unconf_wrong[label] += 1

    return class_unconf_wrong / class_wrong


def unknown_metrics(preds, labels):
    # Unknown metrics
    correct_unknown = 0.
    total_unknown = 0.
    preds_unknown = preds[labels == -1]
    labels_unknown = labels[labels == -1]

    for i in range(len(preds_unknown)):
        pred = preds_unknown[i]
        label = labels_unknown[i]
        # record correctly picked unconfident samples
        if pred == label:
            correct_unknown += 1
        # record all unconfident samples
        total_unknown += 1

    percent_unknown = correct_unknown / total_unknown

    return percent_unknown, total_unknown


def ood_metric(preds, labels, class_counts):
    # f1
    f1 = f_measure(preds, labels)
    # Confident
    class_acc_confident, class_percent_confident, false_pos_percent, total_known = confident_metrics(preds, labels, class_counts)
    # Unconfident
    class_wrong_unconfident = unconfident_metrics(preds, labels)
    # Open
    percent_unknown, total_unknown = unknown_metrics(preds, labels)
    # Confident indices
    conf_preds = np.zeros(len(preds))
    conf_preds[preds != -1] = 1
    return (f1, class_acc_confident, class_percent_confident, false_pos_percent,
            class_wrong_unconfident, percent_unknown, total_unknown, total_known, conf_preds)


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.3, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target, reduction='mean'):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight, reduction=reduction)