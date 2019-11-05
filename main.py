import os
import yaml
from argparse import ArgumentParser

from src.algorithms.utils import get_algorithm

data_root = '/home/zhmiao/datasets/ecology'

parser = ArgumentParser()
parser.add_argument('--config', default='./configs/plain_resnet_041119.yaml')
parser.add_argument('--gpu', default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

with open(args.config) as f:
    config = yaml.load(f)
for k, v in config.items():
    setattr(args, k, v)

alg = get_algorithm(args.algorithm, args)
