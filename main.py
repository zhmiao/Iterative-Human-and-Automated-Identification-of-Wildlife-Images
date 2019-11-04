from src.data.dataloader import get_dataset
from src.models.model import get_model


data_root = '/home/zhmiao/datasets/ecology'

dataset = get_dataset(name='CCT_cis', rootdir=data_root, dset='train')

