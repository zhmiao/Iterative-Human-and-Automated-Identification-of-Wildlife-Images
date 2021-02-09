# %%
import os
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

# %%
spp = ['Ghost', 'Waterbuck', 'Baboon', 'Warthog', 'Bushbuck', 'Impala', 'Oribi', 'Elephant', 'Genet', 'Nyala', 'Setup', 'Bushpig', 'Porcupine', 'Civet', 'Vervet', 'Reedbuck', 'Kudu', 'Buffalo', 'Sable_antelope', 'Duiker_red', 'Hartebeest', 'Wildebeest', 'Guineafowl_helmeted', 'Hare', 'Duiker_common', 'Fire', 'Mongoose_marsh', 'Aardvark', 'Honey_badger', 'Hornbill_ground', 'Mongoose_slender', 'Mongoose_bushy_tailed', 'Samango', 'Mongoose_white_tailed', 'Mongoose_banded', 'Mongoose_large_grey', 'Bushbaby', 'Guineafowl_crested', 'Eland', 'Lion', 'Serval']
conf_acc = [96.597, 91.316, 90.727, 89.446, 87.489, 86.962, 90.348, 96.696, 96.528, 86.420, 96.833, 91.925, 95.767, 95.946, 85.965, 74.834, 85.417, 93.519, 98.734, 98.810, 98.630, 90.476, 86.957, 96.970, 85.294, 100.000, 84.375, 97.143, 100.000, 80.556, 90.000, 96.875, 93.548, 81.481, 61.111, 80.000, 80.769, 100.000, 91.667, 100.000, 75.000]
image_root = '/home/zhmiao/datasets/ecology/Mozambique'

# %%
feats_npz = np.load('../weights/SemiStage2OLTR_Energy/111620_MOZ_PSLABEL_OLTR_Energy_0_feats.npz')
total_file_id = feats_npz['total_file_id']
total_preds = feats_npz['total_preds']
total_energy = feats_npz['total_energy']
total_feats = feats_npz['total_feats']
total_labels = feats_npz['total_labels']

# %%
# total_preds[-total_energy <= 6.77] = -1
the = 6.77
total_feats_norm = normalize(total_feats)
file_id_conf = total_file_id[-total_energy > the]
preds_conf = total_preds[-total_energy > the]
feats_norm_conf = total_feats_norm[-total_energy > the]
labels_conf = total_labels[-total_energy > the]

# %%
pair_dist = pairwise_distances(feats_norm_conf, n_jobs=20)
 
# %%
# spp 1 1482
# spp 34 15

spp_id = 34
target_spp = spp[spp_id]
target_conf_acc = conf_acc[spp_id]
print(target_spp)

# %%
file_id_target = file_id_conf[labels_conf == spp_id]
labels_target = labels_conf[labels_conf == spp_id]
preds_target = preds_conf[labels_conf == spp_id]
dists_target = pair_dist[labels_conf == spp_id]

# np.random.seed(69)

rand_id = np.random.choice(len(file_id_target))
print(rand_id)

rand_id = 15

file_id_anchor = file_id_target[rand_id]
label_anchor = labels_target[rand_id]
pred_anchor = preds_target[rand_id]
dists_anchor = dists_target[rand_id]
id_ret = dists_anchor.argsort()[1:6]
file_id_ret = file_id_conf[id_ret]
labels_ret = labels_conf[id_ret]
preds_ret = preds_conf[id_ret]

# %%
# im_list = []
# im_list.append(Image.open(os.path.join(image_root, file_id_anchor)).crop((50, 50, 200, 200)).resize((256, 256)))
# im_list.append(Image.open(os.path.join(image_root, file_id_ret[0])).crop((20, 50, 150, 180)).resize((256, 256)))
# im_list.append(Image.open(os.path.join(image_root, file_id_ret[1])).crop((80, 80, 200, 200)).resize((256, 256)))
# im_list.append(Image.open(os.path.join(image_root, file_id_ret[2])).crop((50, 50, 180, 180)).resize((256, 256)))
# im_list.append(Image.open(os.path.join(image_root, file_id_ret[3])).crop((50, 50, 180, 180)).resize((256, 256)))
# im_list.append(Image.open(os.path.join(image_root, file_id_ret[4])).crop((140, 100, 240, 200)).resize((256, 256)))

# %%
grid_im = Image.new('RGB', (256 * 6 + 20, 256 + 48), color='black')

draw = ImageDraw.Draw(grid_im)
font = ImageFont.truetype('./arial.ttf', size=30)

idx = 0
for i in range(0, 256 * 6, 256):
    
    l = int(i/256) * 4
    
    if idx == 0:
        im = Image.open(os.path.join(image_root, file_id_anchor))
    else:
        im = Image.open(os.path.join(image_root, file_id_ret[idx - 1]))
    # im = im_list[idx]
    # if idx > 0:
    #     pred_ret = preds_ret[idx - 1]
    
    grid_im.paste(im, (i + l, 0))

    if idx > 0:
        color = 'green' if pred_ret == label_anchor else 'red'
        draw.rectangle(((i + l, 0), (i + l + 256, 256)), outline=color, width=10)

    idx += 1
    draw.text((6, 256 + 6), '{} (high-confidence acc.: {:.1f})'.format(target_spp, target_conf_acc), 'yellow', font=font)

# %%
grid_im
# %%
grid_im.save('./Figs/retrieval/{}_{}_{}.png'.format(target_spp, spp_id, rand_id))
# %%
