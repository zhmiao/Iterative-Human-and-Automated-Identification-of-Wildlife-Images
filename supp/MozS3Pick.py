# %% codecell
import os
import numpy as np
from tqdm import tqdm
from shutil import copyfile

# %% codecell
class_indices_S2_rev = {
    'Waterbuck': 0,
    'Baboon': 1,
    'Warthog': 2,
    'Bushbuck': 3,
    'Impala': 4,
    'Oribi': 5,
    'Elephant': 6,
    'Nyala': 7,
    'Genet': 8,
    'Civet': 9,
    'Vervet': 10,
    'Bushpig': 11,
    'Reedbuck': 12,
    'Kudu': 13,
    'Porcupine': 14,
    'Buffalo': 15,
    'Sable_antelope': 16,
    'Duiker_red': 17,
    'Wildebeest': 18,
    'Hartebeest': 19,
    'Guineafowl_helmeted': 20,
    'Hare': 21,
    'Duiker_common': 22,
    'Mongoose_marsh': 23,
    'Aardvark': 24,
    'Honey_badger': 25,
    'Hornbill_ground': 26,
    'Mongoose_slender': 27,
    'Mongoose_bushy_tailed': 28
}

class_indices_S2 = {class_indices_S2_rev[k]: k for k in class_indices_S2_rev}

# %% codecell

root = '/home/zhmiao/datasets/ecology/GNP'

confident_path = '/home/zhmiao/repos/AnimalActiveLearing_srv/weights/GTPSMemoryStage2_ConfPseu/051620_MOZ_S2_0_preds_conf.txt'

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
save_root = os.path.join(root, 'S3_pickout')
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
