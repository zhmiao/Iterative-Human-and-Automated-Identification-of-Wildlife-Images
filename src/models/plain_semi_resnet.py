import torch.nn as nn
import torch.nn.functional as F

from models.plain_resnet import PlainResNetClassifier

from .utils import register_model


@register_model('PlainSemiResNetClassifier')
class PlainSemiResNetClassifier(PlainResNetClassifier):

    name = 'PlainSemiResNetClassifier'

    def __init__(self, num_cls=10, weights_init='ImageNet', num_layers=18, init_feat_only=True, T=None, alpha=None):
        self.T = T
        self.alpha = alpha
        super(PlainSemiResNetClassifier, self).__init__(num_cls=num_cls, weights_init=weights_init,
                                                        num_layers=num_layers, init_feat_only=init_feat_only)

    def setup_critera(self):
        self.criterion_cls_hard = nn.CrossEntropyLoss()
        self.criterion_cls_soft = self._make_distill_criterion(alpha=self.alpha, T=self.T)

    @staticmethod
    def _make_distill_criterion(alpha=0.5, T=4.0):
        print('** alpha is {} and temperature is {} **\n'.format(alpha, T))
        def criterion(outputs, labels, targets):
            # Soft cross entropy
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)

            # _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
            _soft_loss = nn.KLDivLoss()(_p, _q)

            # Soft hard combination
            _soft_loss = _soft_loss * T * T
            _hard_loss = F.cross_entropy(outputs, labels)
            loss = alpha * _soft_loss + (1. - alpha) * _hard_loss
            return loss
        return criterion

