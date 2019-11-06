import os
import shutil
import yaml
from  datetime import datetime
import logging
from argparse import ArgumentParser

from src.algorithms.utils import get_algorithm

###################
# Parse arguments #
###################
parser = ArgumentParser()
parser.add_argument('--config', default='./configs/plain_resnet_041119.yaml',
                    help='Configuration path.')
parser.add_argument('--session', default=0,
                    help='Session id.')
parser.add_argument('--gpu', default='0',
                    help='GPU id.')
parser.add_argument('--evaluate', default=False, action='store_true',
                    help='If evaluate the model.')
args = parser.parse_args()

###########
# Set gpu #
###########
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


###############################
# Load configurations to args #
###############################
with open(args.config) as f:
    config = yaml.load(f)
for k, v in config.items():
    setattr(args, k, v)

#################
# Create logger #
#################
log_root = './log'
os.makedirs(log_root, exist_ok=True)
log_file = os.path.join(log_root, '{}_{}_{}.log'.format(args.algorithm, args.conf_id, args.session))
logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger('MAIN')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_file, mode='a' if args.evaluate else 'w')
logger.addHandler(handler)
setattr(args, 'logger', logger)

##############
# Algorithms #
##############
alg = get_algorithm(args.algorithm, args)
if not args.evaluate:
    alg.train()

