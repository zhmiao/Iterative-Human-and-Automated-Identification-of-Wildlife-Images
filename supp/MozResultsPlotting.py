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
ax.set_title('Update Performance Comparison', fontsize=35, y=1.01)
ax.set_xlabel('Acc. (%)', fontsize=25, y=-0.02)
plt.savefig('./Figs/Group2UpdatePerformance.png', bbox_inches='tight')

# %%
ft_acc = [96.2, 88.8, 87.3, 87.4, 84.5, 84.0, 83.8, 88.2, 85.5, 73.9, 87.4, 83.1, 83.9, 82.9, 73.2, 65.8, 70.9, 89.0, 85.2, 86.8, 91.2, 83.5, 64.6, 78.8, 62.7, 100.0, 70.6, 77.6, 60.3, 80.0, 68.0, 74.0, 58.0, 52.0, 38.0, 44.0, 36.0, 95.0, 90.0, 70.0, 45.0]
ours_acc = [90.180, 82.430, 81.079, 79.741, 72.347, 77.082, 76.711, 85.075, 84.048, 75.124, 85.965, 83.051, 88.261, 83.886, 80.976, 75.263, 77.215, 84.828, 86.111, 89.623, 84.615, 82.418, 74.390, 80.769, 80.392, 100.000, 71.765, 81.034, 63.793, 72.000, 72.000, 72.000, 70.000, 64.000, 52.000, 48.000, 50.000, 100.000, 70.000, 75.000, 60.000]
class_count = [20500, 17938, 15660, 17400, 6622, 7153, 3832, 2471, 1976, 1569, 1229, 1040, 1152, 699, 739, 740, 556, 479, 323, 370, 394, 303, 304, 214, 194, 160, 343, 235, 234, 203, 165, 161, 99, 84, 70, 63, 39, 46, 44, 42, 41]
ours_ann = [4248, 2079, 2335, 4224, 2179, 1306, 966, 470, 888, 434, 389, 377, 300, 123, 263, 203, 161, 63, 48, 116, 63, 44, 250, 166, 92, 14, 287, 128, 190, 161, 157, 106, 48, 79, 62, 54, 31, 35, 31, 32, 32]
spp = ['Ghost', 'Waterbuck', 'Baboon', 'Warthog', 'Bushbuck', 'Impala', 'Oribi', 'Elephant', 'Genet', 'Nyala', 'Setup', 'Bushpig', 'Porcupine', 'Civet', 'Vervet', 'Reedbuck', 'Kudu', 'Buffalo', 'Sable_antelope', 'Duiker_red', 'Hartebeest', 'Wildebeest', 'Guineafowl_helmeted', 'Hare', 'Duiker_common', 'Fire', 'Mongoose_marsh', 'Aardvark', 'Honey_badger', 'Hornbill_ground', 'Mongoose_slender', 'Mongoose_bushy_tailed', 'Samango', 'Mongoose_white_tailed', 'Mongoose_banded', 'Mongoose_large_grey', 'Bushbaby', 'Guineafowl_crested', 'Eland', 'Lion', 'Serval']
# %%
ft_acc = np.array(ft_acc)
ours_acc = np.array(ours_acc)
class_count = np.array(class_count)
ours_ann = np.array(ours_ann)
ours_percent = ours_ann / class_count
ours_eff = ours_acc / ours_percent
spp = np.array(spp)
# acc_diff = ours_acc - ft_acc

# %%
plt.style.use('seaborn-whitegrid')

fig, ax1 = plt.subplots(figsize=(30, 10))

ax1.axvspan(-0.5, 25.505,
            facecolor='blue', alpha=0.07)
ax1.axvspan(25.505, 40.5,
            facecolor='lightcoral', alpha=0.2)

lw = 5
a = 0.7
line_ct = ax1.plot(range(41), sorted(class_count, reverse=True), 'C5', alpha=a, linewidth=lw)

line_ann = ax1.plot(range(41), sorted(ours_ann, reverse=True), 'C2', alpha=a, linewidth=lw)

ax1.set_ylabel('# of human annotations per-class', color='C3', labelpad=8, fontsize=40)
ax1.tick_params('y', colors='C3', labelsize=30)
ax1.tick_params('x', labelsize=20)
# ax1.set_yticks(range(0, 21000, 5000))
ax1.set_yscale('log')
ax1.set_xticks(range(0, 41))
ax1.set_xticklabels(spp, fontsize=32)
# ax1.set_ylim((-1000, 21000))
for label in ax1.get_xmajorticklabels():
    label.set_rotation(50)
    label.set_horizontalalignment("right")
    label.set_rotation_mode("anchor")

ax2 = ax1.twinx()
w = 0.4
a = 0.7
bar_plot_1 = ax2.bar(np.arange(len(ours_eff)) - w/2, ours_eff, width=w, alpha=a)
bar_plot_2 = ax2.bar(np.arange(len(ft_acc)) + w/2, ft_acc, width=w, alpha=a)

leg_list = []
leg_list.append(bar_plot_1)
leg_list.append(bar_plot_2)
leg_list.append(line_ct[0])
leg_list.append(line_ann[0])

ax2.set_xlabel('Classes')
ax2.set_ylabel('Efficiency', color='C0', labelpad=11, fontsize=40)
ax2.tick_params('y', colors='C0', labelsize=30)

ax2.set_ylim((-0.4, 1000))
ax2.set_yticks([0.0, 300, 600, 900])

ax2.spines['right'].set_color('C0')
ax2.spines['left'].set_color('C3')
ax2.grid(None)

ax1.text(9, 8000, 'In both Group 1 & 2', fontsize=36, color='royalblue')
ax1.text(30.5, 8000, 'Only in Group 2', fontsize=36, color='royalblue')

fig.legend(leg_list, ['Our framework', 
                      'Full annotation transfer learning',
                      'Full human annotation counts',
                      'Actual human annotation used in our framework'],
           fontsize=25, loc=(0.62, 0.57))

fig.tight_layout()

plt.savefig('./Figs/efficiency.pdf', format='pdf', bbox_inches='tight')

# plt.show()
# %%
