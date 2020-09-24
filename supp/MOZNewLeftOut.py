# %% codecell
import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image

# %% codecell
root = '/home/zhmiao/datasets/ecology/Mozambique/'

leftout_new_list = open(os.path.join(root, 'SplitLists', 'leftout_mix_new.txt'), 'w')

with open(os.path.join(root, 'SplitLists', 'leftout_mix.txt'), 'r') as f:
    for line in tqdm(f) :
        line_sp = line.replace('\n', '').split(' ')
        # if line_sp[1] != 'Setup' and line_sp[1] != 'Fire' and os.path.exists(os.path.join(root, line_sp[0])):
        if os.path.exists(os.path.join(root, line_sp[0])):
            leftout_new_list.write(line)

leftout_new_list.close()
