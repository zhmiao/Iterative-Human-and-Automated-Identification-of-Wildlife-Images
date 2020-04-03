import os
import copy
import math
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.autograd.function import Function

from .utils import register_model, BaseModule
from .resnet_backbone import ResNetFeature, BasicBlock, Bottleneck, model_urls


@register_model('MemoryResNetClassifier')
class MemoryResNetClassifier(BaseModule):

    name = 'MemoryResNetClassifier'

    def __init__(self, num_cls=10, weights_init='ImageNet', num_layers=18, init_feat_only=True):
        super(MemoryResNetClassifier, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.feature = None
        self.fc_hallucinator = None
        self.fc_selector = None
        self.cosnorm_classifier = None
        self.criterion_cls = None
        self.criterion_ctr = None
        self.best_weights = None
        self.feature_dim = None

        # Model setup and weights initialization
        self.setup_net()

        if weights_init == 'ImageNet':
            self.load(model_urls['resnet{}'.format(num_layers)], feat_only=init_feat_only)
        elif os.path.exists(weights_init):
            self.load(weights_init, feat_only=init_feat_only)
        elif weights_init != 'ImageNet' and not os.path.exists(weights_init):
            raise NameError('Initial weights not exists {}.'.format(weights_init))

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

        self.feature_dim = 512 * block.expansion

        self.fc_hallucinator = nn.Linear(self.feature_dim, self.num_cls)
        self.fc_selector = nn.Linear(self.feature_dim, self.feature_dim)
        self.cosnorm_classifier = CosNorm_Classifier(self.feature_dim, self.feature_dim)

    def setup_critera(self):
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_ctr = DiscCentroidsLoss(self.num_cls, self.feature_dim)

    def load(self, init_path, feat_only=False):

        if 'http' in init_path:
            init_weights = load_state_dict_from_url(init_path, progress=True)
        else:
            init_weights = torch.load(init_path)

        if feat_only:
            init_weights = OrderedDict({k.replace('feature.', ''): init_weights[k] for k in init_weights})
            self.feature.load_state_dict(init_weights, strict=False)
            load_keys = set(init_weights.keys())
            self_keys = set(self.feature.state_dict().keys())
        else:
            self.load_state_dict(init_weights, strict=False)
            load_keys = set(init_weights.keys())
            self_keys = set(self.state_dict().keys())

        missing_keys = self_keys - load_keys
        unused_keys = load_keys - self_keys
        print("missing keys: {}".format(sorted(list(missing_keys))))
        print("unused_keys: {}".format(sorted(list(unused_keys))))

    def save(self, out_path):
        torch.save(self.best_weights, out_path)

    def update_best(self):
        self.best_weights = copy.deepcopy(self.state_dict())


class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())


class DiscCentroidsLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(DiscCentroidsLoss, self).__init__()
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.disccentroidslossfunc = DiscCentroidsLossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):

        batch_size = feat.size(0)

        #############################
        # calculate attracting loss #
        #############################

        feat = feat.view(batch_size, -1)

        # To check the dim of centroids and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss_attract = self.disccentroidslossfunc(feat.clone(), label, self.centroids.clone(), batch_size_tensor).squeeze()

        ############################
        # calculate repelling loss #
        #############################

        distmat = torch.pow(feat.clone(), 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centroids.clone(), 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, feat.clone(), self.centroids.clone().t())

        classes = torch.arange(self.num_classes).long().cuda()
        labels_expand = label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        distmat_neg = distmat
        distmat_neg[mask] = 0.0
        margin = 10.0
        loss_repel = torch.clamp(margin - distmat_neg.sum() / (batch_size * self.num_classes), 0.0, 1e6)

        loss = loss_attract + 0.01 * loss_repel

        return loss


class DiscCentroidsLossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centroids, batch_size):
        ctx.save_for_backward(feature, label, centroids, batch_size)
        centroids_batch = centroids.index_select(0, label.long())
        return (feature - centroids_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centroids, batch_size = ctx.saved_tensors
        centroids_batch = centroids.index_select(0, label.long())
        diff = centroids_batch - feature
        # init every iteration
        counts = centroids.new_ones(centroids.size(0))
        ones = centroids.new_ones(label.size(0))
        grad_centroids = centroids.new_zeros(centroids.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centroids.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centroids = grad_centroids/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centroids / batch_size, None



