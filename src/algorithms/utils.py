import numpy as np


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

    def save_model(self):
        pass


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


def confident_metrics(preds, labels, unique_classes, class_counts):
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

    return class_acc_confident, class_percent_confident, false_pos_percent


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

    return percent_unknown


def stage_1_metric(preds, labels, unique_classes, class_counts):
    # f1
    f1 = f_measure(preds, labels)
    # Confident
    class_acc_confident, class_percent_confident, false_pos_percent = confident_metrics(preds, labels, unique_classes, class_counts)
    # Open
    percent_unknown = unknown_metrics(preds, labels)
    # Confident indices
    conf_preds = np.zeros(len(preds))
    conf_preds[preds != -1] = 1
    return f1, class_acc_confident, class_percent_confident, false_pos_percent, percent_unknown, conf_preds



