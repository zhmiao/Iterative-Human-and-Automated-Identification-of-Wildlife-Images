import os
import yaml
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
parser.add_argument('--gpu', default=0,
                    help='GPU id.')
parser.add_argument('--np_threads', default=4,
                    help='Num of threads of numpy.')
parser.add_argument('--evaluate', default=False, action='store_true',
                    help='If evaluate the model.')
parser.add_argument('--deploy', default=False, action='store_true',
                    help='Actual model deployment.')
parser.add_argument('--deploy_ood', default=False, action='store_true',
                    help='Actual model deployment for out of distribution.')
parser.add_argument('--ood_ft', default=False, action='store_true',
                    help='Mode for fine-tuning ood.')
args = parser.parse_args()

#############################
# Set environment variables #
#############################
# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# Set numpy threads
os.environ["OMP_NUM_THREADS"] = str(args.np_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.np_threads)
os.environ["MKL_NUM_THREADS"] = str(args.np_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.np_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.np_threads)

###############################
# Load configurations to args #
###############################
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
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
if args.evaluate:
    alg.set_eval()
    alg.logger.info('\nTesting...')
    _ = alg.evaluate(loader=alg.testloader)
    # _ = alg.evaluate(loader=alg.valloader)
elif args.deploy:
    alg.set_eval()
    alg.logger.info('\nDeploying...')
    alg.deploy(loader=alg.deployloader)
elif args.deploy_ood:
    alg.set_eval()
    alg.logger.info('\nDeploying OOD...')
    alg.deploy_ood(loader=alg.deployloader_ood)
elif args.ood_ft:
    alg.set_eval()
    alg.logger.info('\nFine-tuning OOD...')
    alg.ood_ft()
else:
    alg.set_train()
    alg.logger.info('\nTraining...')
    alg.train()

