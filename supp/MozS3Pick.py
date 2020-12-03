# %% codecell
import os
import numpy as np
from tqdm import tqdm
from shutil import copyfile

# %% codecell
class_indices_S2_rev = {
    'Waterbuck': 1,
    'Baboon': 2,
    'Warthog': 3,
    'Bushbuck': 4,
    'Impala': 5,
    'Oribi': 6,
    'Elephant': 7,
    'Genet': 8,
    'Nyala': 9,
    'Setup': 10,
    'Bushpig': 11,
    'Porcupine': 12,
    'Civet': 13,
    'Vervet': 14,
    'Reedbuck': 15,
    'Kudu': 16,
    'Buffalo': 17,
    'Sable_antelope': 18,
    'Duiker_red': 19,
    'Hartebeest': 20,
    'Wildebeest': 21,
    'Guineafowl_helmeted': 22,
    'Hare': 23,
    'Duiker_common': 24,
    'Fire': 25,
    'Mongoose_marsh': 26,
    'Aardvark': 27,
    'Honey_badger': 28,
    'Hornbill_ground': 29,
    'Mongoose_slender': 30,
    'Mongoose_bushy_tailed': 31,
    'Samango': 32,
    'Mongoose_white_tailed': 33,
    'Mongoose_banded': 34,
    'Mongoose_large_grey': 35,
    'Bushbaby': 36,
    'Guineafowl_crested': 37,
    'Eland': 38,
    'Lion': 39,
    'Serval': 40
}

class_indices_S2 = {class_indices_S2_rev[k]: k for k in class_indices_S2_rev}

# %% codecell

root = '/home/zhmiao/datasets/ecology/GNP'

# confident_path = '/home/zhmiao/repos/AnimalActiveLearing_srv/weights/GTPSMemoryStage2_ConfPseu/051620_MOZ_S2_0_preds_conf.txt'
# confident_path = '/home/zhmiao/repos/AnimalActiveLearing_srv/weights/GTPSMemoryStage2_ConfPseu_SoftIter/072520_MOZ_S2_soft_iter_0_preds_conf.txt'
confident_path = '/home/zhmiao/repos/AnimalActiveLearning/weights/SemiStage2OLTR_Energy/111620_MOZ_PSLABEL_OLTR_Energy_0_preds_conf.txt'

# %% codecell

f = open(confident_path, 'r')

file_id_list = []
cat_list = []

for line in tqdm(f):
    line_sp = line.replace('\n', '').rsplit(' ', 1)
    file_id = line_sp[0]
    cat = class_indices_S2[int(line_sp[1])]
    file_id_list.append(file_id)
    cat_list.append(cat)


f.close()

# %% codecell
file_id_list = np.array(file_id_list)
cat_list = np.array(cat_list)

# %% codecell

np.random.seed(10)
rand_idx = np.random.choice(range(len(cat_list)), 1000)

file_id_sel = file_id_list[rand_idx]
cat_sel = cat_list[rand_idx]

# %% codecell
save_root = os.path.join(root, 'S3_pickout_soft_iter_120220')
os.makedirs(save_root, exist_ok=True)

# %% codecell
for file_id, cat in tqdm(zip(file_id_sel, cat_sel)):

    from_path = os.path.join(root, file_id)

    file_id = file_id.replace('/', ':::')

    save_path = os.path.join(save_root, file_id)

    if '.JPG' in save_path:
        save_path = save_path.replace('.JPG', '_{}.JPG'.format(cat))
    elif '.jpg' in save_path:
        save_path = save_path.replace('.jpg', '_{}.jpg'.format(cat))

    copyfile(from_path, save_path)

# %%
