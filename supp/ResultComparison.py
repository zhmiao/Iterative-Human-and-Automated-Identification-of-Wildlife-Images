# %%
import os
import random
import numpy as np
from numpy.lib.shape_base import expand_dims
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
validation_path = '/home/zhmiao/repos/AnimalActiveLearning/weights/SemiStage2OLTR_Energy/111620_MOZ_PSLABEL_OLTR_Energy_0_preds_eval.txt'


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
class_indices_S2[0] = 'Ghost'
def load_list_eval(path):
    with open(path, 'r') as f:
        file_id_list = []
        pred_list = []
        cat_list = []
        for line in tqdm(f):
            line_sp = line.replace('\n', '').rsplit(' ', 2)
            file_id = line_sp[0]
            pred = -1 if int(line_sp[1]) == -1 else class_indices_S2[int(line_sp[1])]
            cat = -1 if int(line_sp[2]) == -1 else class_indices_S2[int(line_sp[2])]
            file_id_list.append(file_id)
            pred_list.append(pred)
            cat_list.append(cat)
    return (np.array(file_id_list), np.array(cat_list), np.array(pred_list))
# %%
file_id, preds, cats = load_list_eval(validation_path)

# %%
image_root = '/home/zhmiao/datasets/ecology/Mozambique'
wrong_ids = file_id[preds != cats]
wrong_preds = preds[preds != cats]
wrong_cats = cats[preds != cats]
correct_ids = file_id[preds == cats]
correct_preds = preds[preds == cats]

# %%
seed = random.choice(list(range(0, 50000)))
# seed = 5742
random.seed(seed)
wrong_index = random.choice(list(range(len(wrong_ids))))
correct_index = random.choice(list(range(len(correct_ids))))

wrong = Image.open(os.path.join(image_root, wrong_ids[wrong_index]))
wrong_p = wrong_preds[wrong_index]
wrong_cat = wrong_cats[wrong_index]
correct = Image.open(os.path.join(image_root, correct_ids[correct_index]))
correct_p = correct_preds[correct_index]

# %%
print(wrong_p)
print(wrong_cat)
wrong

# %%
print(correct_p)
correct

# %%
wrong.save('./Figs/wrong_correct/{}_{}_{}_{}_{}'
           .format(seed, 'wrong', wrong_p, wrong_cat,
                   wrong_ids[wrong_index].replace('/', ':::').replace('.JPG', '.PNG')))

# %%
correct.save('./Figs/wrong_correct/{}_{}_{}'
             .format(seed, 'correct', correct_p, 
                     correct_ids[correct_index].replace('/', ':::').replace('.JPG', '.PNG')))

