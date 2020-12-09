# %%
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

# %%
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

# %%
image_root = '/home/zhmiao/datasets/ecology/Mozambique/Mozambique_season_3'
confident_path = '/home/zhmiao/repos/AnimalActiveLearning/weights/SemiStage2OLTR_Energy/111620_MOZ_PSLABEL_OLTR_Energy_0_preds_conf.txt'
unconfident_path = '/home/zhmiao/repos/AnimalActiveLearning/weights/SemiStage2OLTR_Energy/111620_MOZ_PSLABEL_OLTR_Energy_0_preds_unconf.txt'

# %%
def load_list(path):
    with open(path, 'r') as f:
        file_id_list = []
        cat_list = []
        for line in tqdm(f):
            line_sp = line.replace('\n', '').rsplit(' ', 1)
            file_id = line_sp[0]
            cat = class_indices_S2[int(line_sp[1])]
            file_id_list.append(file_id)
            cat_list.append(cat)
    return (np.array(file_id_list), np.array(cat_list))

# %%
file_id_conf, cat_conf = load_list(confident_path)
file_id_unconf, cat_unconf = load_list(unconfident_path)

# %%
seed = random.choice(list(range(0, 50000)))
random.seed(seed)
target_spp = random.choice(list(class_indices_S2.values()))
conf_id = random.choice(file_id_conf[cat_conf == target_spp])
unconf_id = random.choice(file_id_unconf[cat_unconf == target_spp])
conf = Image.open(os.path.join(image_root, conf_id))
unconf = Image.open(os.path.join(image_root, unconf_id))

# %%
target_spp
# %%
conf
# %%
unconf
# %%
conf.save('./Figs/conf_unconf/{}_{}_{}_{}'.format(seed, 'conf', target_spp, conf_id.replace('/', ':::').replace('.JPG', '.PNG')))
unconf.save('./Figs/conf_unconf/{}_{}_{}_{}'.format(seed, 'unconf', target_spp, unconf_id.replace('/', ':::').replace('.JPG', '.PNG')))

# %%
file_id_conf[0]

# %%
