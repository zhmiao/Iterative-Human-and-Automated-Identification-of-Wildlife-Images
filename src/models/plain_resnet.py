import os
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.hub import load_state_dict_from_url

from .utils import register_model, BaseModule
from .resnet_backbone import ResNetFeature, BasicBlock, Bottleneck, model_urls


@register_model('PlainResNetClassifier')
class PlainResNetClassifier(BaseModule):

    name = 'PlainResNetClassifier'

    def __init__(self, num_cls=10, weights_init='ImageNet', num_layers=18, init_feat_only=True,
                 parallel=False, norm=False):
        super(PlainResNetClassifier, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.feature = None
        self.classifier = None
        self.criterion_cls = None
        self.best_weights = None
        self.feature_dim = None
        self.norm = norm

        # Model setup and weights initialization
        self.setup_net()

        if weights_init == 'ImageNet':
            self.load(model_urls['resnet{}'.format(num_layers)], feat_only=init_feat_only)
        elif os.path.exists(weights_init):
            self.load(weights_init, feat_only=init_feat_only)
        elif weights_init != 'ImageNet' and not os.path.exists(weights_init):
            raise NameError('Initial weights not exists {}.'.format(weights_init))
        
        if parallel:
            print('**USING DATAPARALLEL**')
            self.feature = nn.DataParallel(self.feature)
            self.classifier = nn.DataParallel(self.classifier)
        
        # Criteria setup
        self.setup_critera()

    def setup_net(self):

        kwargs = {}

        if self.num_layers == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif self.num_layers == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif self.num_layers == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
        else:
            raise Exception('ResNet Type not supported.')

        self.feature = ResNetFeature(block, layers, **kwargs)
        if self.norm:
            print('**USING NORMALIZED CLASSIFIER**')
            self.classifier = NormedLinear(512 * block.expansion, self.num_cls)
        else:
            self.classifier = nn.Linear(512 * block.expansion, self.num_cls)
        self.feature_dim = 512 * block.expansion

    def setup_critera(self):
        self.criterion_cls = nn.CrossEntropyLoss()

    def load(self, init_path, feat_only=False):

        if 'http' in init_path:
            init_weights = load_state_dict_from_url(init_path, progress=True)
        else:
            init_weights = torch.load(init_path)

        if feat_only:
            init_weights = OrderedDict({k.replace('module.', '').replace('feature.', ''): init_weights[k] for k in init_weights})
            self.feature.load_state_dict(init_weights, strict=False)
            load_keys = set(init_weights.keys())
            self_keys = set(self.feature.state_dict().keys())
        else:
            init_weights = OrderedDict({k.replace('module.', ''): init_weights[k] for k in init_weights})
            self.load_state_dict(init_weights, strict=False)
            load_keys = set(init_weights.keys())
            self_keys = set(self.state_dict().keys())

        missing_keys = self_keys - load_keys
        unused_keys = load_keys - self_keys
        print("missing keys: {}".format(sorted(list(missing_keys))))
        print("unused_keys: {}".format(sorted(list(unused_keys))))

    def save(self, out_path):
        torch.save(self.best_weights, out_path)
        torch.save(self.state_dict, out_path.replace('.pth', '_final.pth'))

    def update_best(self):
        self.best_weights = copy.deepcopy(self.state_dict())


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out