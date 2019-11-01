from src.data.dataloader import get_dataset

data_root = '/home/zhmiao/datasets/ecology'

dataset = get_dataset(name='CCT_cis', rootdir=data_root, dset='train')

