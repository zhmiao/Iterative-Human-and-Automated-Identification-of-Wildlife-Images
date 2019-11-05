import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from .utils import register_model
from .resnet_backbone import ResNetFeature, BasicBlock, Bottleneck, model_urls


@register_model('PlainResNetClassifier')
class PlainResNetClassifier(nn.Module):

    name = 'PlainResNetClassifier'

    def __init__(self, num_cls=10, weights_init='ImageNet', num_layers=18):
        super(PlainResNetClassifier, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.feature = None
        self.classifier = None
        self.criterion_cls = None

        # Model setup and weights initialization
        self.setup_net()
        if weights_init == 'ImageNet':
            self.load(model_urls['resnet{}'.format(num_layers)])
        else:
            raise Exception('Pretrained weights not supported.')

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
        self.classifier = nn.Linear(512 * block.expansion, self.num_cls)

    def setup_critera(self):
        self.criterion_cls = nn.CrossEntropyLoss()

    def load(self, init_path):

        if 'http' in init_path:
            init_weights = load_state_dict_from_url(init_path, progress=True)
        else:
            init_weights = torch.load(init_path)

        self.load_state_dict(init_weights, strict=False)

        load_keys = set(init_weights.keys())
        self_keys = set(self.state_dict().keys())
        missing_keys = self_keys - load_keys
        unused_keys = load_keys - self_keys
        print("missing keys: {}".format(sorted(list(missing_keys))))
        print("unused_keys: {}".format(sorted(list(unused_keys))))

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

