# %%
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

# %%
root = '/home/zhmiao/datasets/ecology/Mozambique/'

group_1_list = []

with open(os.path.join(root, 'SplitLists', 'train_mix_season_1_lt.txt'), 'r') as f:
        for line in tqdm(f):
            line = line.replace('\n', '')
            line_sp = line.split(' ')
            label = line_sp[1]
            group_1_list.append(label)

group_2_list = []

with open(os.path.join(root, 'SplitLists', 'train_mix_season_2_lt.txt'), 'r') as f:
        for line in tqdm(f):
            line = line.replace('\n', '')
            line_sp = line.split(' ')
            label = line_sp[1]
            group_2_list.append(label)

# %%
group_1_counts = {cat.lower(): count
                  for cat, count in zip(*np.unique(group_1_list, return_counts=True))}

group_2_counts = {cat.lower(): count
                  for cat, count in zip(*np.unique(group_2_list, return_counts=True))}


# %%
plain_group_1 = pd.read_csv('./CSVs/PlainGroup1.csv')
energy_group_1 = pd.read_csv('./CSVs/EnergyGroup1.csv')
ft_full_group_2 = pd.read_csv('./CSVs/FTFullGroup2.csv')
ft_gt_group_2 = pd.read_csv('./CSVs/FTGTGroup2.csv')
semi_oltr_group_2 = pd.read_csv('./CSVs/SemiOLTRGroup2.csv')
semi_oltr_energy_group_2 = pd.read_csv('./CSVs/SemiOLTREnergyGroup2.csv')

# %%
##########
# Group 1 Energy
##########
cat = [c.split('(')[1].split(')')[0].lower() for c in list(energy_group_1['Categories'])]
ct = [group_1_counts[c] for c in cat]
cat = [c + ' ({})'.format(n) for c, n in zip(cat, ct)]
conf_acc = list(energy_group_1['Conf. Acc'])

# %%
sns.set(font_scale=2.5, style='white')
plt.subplots(figsize=(15, 30))
ax = sns.barplot(x=conf_acc, y=cat, dodge=False, color='C0')
for p in ax.patches:
    ax.annotate(format(p.get_width(), '.1f'), 
                   (p.get_width() + 8, p.get_y() + p.get_height() / 2 + 0.2), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
# ax.set_title('Per-category Confident Acc. (Group 1)', fontsize=50, y=1.01)
ax.set_xlabel('Acc. (%)', fontsize=40, y=-0.02)
ax.set_xlim((0, 119))
plt.savefig('./Figs/Group1ConfAcc.png', bbox_inches='tight')

# %%
##########
# Group 2 Energy
##########
cat = [c.split('(')[1].split(')')[0].lower() for c in list(semi_oltr_energy_group_2['Categories'])]
for i, c in enumerate(cat):
    if c == 'guineagowl_crested':
        cat[i] = 'guineafowl_crested'
ct = [group_2_counts[c] for c in cat]
cat = [c + ' ({})'.format(n) for c, n in zip(cat, ct)]
conf_acc = list(semi_oltr_energy_group_2['Conf. Acc'])

# %%
sns.set(font_scale=2.5, style='white')
plt.subplots(figsize=(15, 30))
ax = sns.barplot(x=conf_acc, y=cat, dodge=False, color='C0')
for p in ax.patches:
    ax.annotate(format(p.get_width(), '.1f'), 
                   (p.get_width() + 8, p.get_y() + p.get_height() / 2 + 0.2), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
# ax.set_title('Per-category Confident Acc. (Group 1)', fontsize=50, y=1.01)
ax.set_xlabel('Acc. (%)', fontsize=40, y=-0.02)
ax.set_xlim((0, 119))
plt.savefig('./Figs/Group2ConfAcc.png', bbox_inches='tight')

# %%
##########
# Group 2 update
##########
cat = [c.split('(')[1].split(')')[0].lower() for c in list(semi_oltr_energy_group_2['Categories'])]
for i, c in enumerate(cat):
    if c == 'guineagowl_crested':
        cat[i] = 'guineafowl_crested'

cat = cat + cat + cat

# cat = [0 for _ in range(len(ft_full_acc) * 3)]

ft_full_acc = ft_full_group_2['Eval Acc'] 
ft_gt_acc = ft_gt_group_2['Eval Acc'] 
semi_oltr_acc = semi_oltr_group_2['Eval Acc'] 

acc = list(ft_full_acc) + list(ft_gt_acc) + list(semi_oltr_acc)

hue = (['FineTune Full Ann.' for _ in range(len(ft_full_acc))]
       + ['FineTune Human Ann. Only' for _ in range(len(ft_full_acc))]
       + ['Semi-OLTR' for _ in range(len(ft_full_acc))])

# %%
sns.set(font_scale=2, style='whitegrid')
plt.subplots(figsize=(15, 30))
ax = sns.barplot(x=acc, y=cat, hue=hue, dodge=True)
ax.set_title('Distribution of Group 1 & 2', fontsize=35, y=1.01)
ax.set_xlabel('Total #', fontsize=25, y=-0.02)
plt.savefig('./Figs/Group2UpdatePerformance.png', bbox_inches='tight')

# %%
