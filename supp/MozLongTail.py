# %% codecell
import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# %% codecell
rename = {
    'Baboon_': 'Baboon',
    'Elephant_': 'Elephant',
    'Honey_badger_': 'Honey_badger',
    'Honey_Badger': 'Honey_badger',
    'Honey Badger': 'Honey_badger',
    'Mongoose_larger_gray': 'Mongoose_large_grey',
    'Mongoose_large_gray': 'Mongoose_large_grey',
    'Mongoose_white tailed': 'Mongoose_white_tailed',
    'Hippopotamus': 'Hippo',
    'Sable': 'Sable_antelope',
    'Ground_hornbill': 'Hornbill_ground',
    'Human': 'Setup',
}

# %% codecell
root = '/home/zhmiao/datasets/ecology/Mozambique/'
s1_ori = os.path.join(root, 'SplitLists', 'Mozambique_season_1_all.txt.ori')
s2_ori = os.path.join(root, 'SplitLists', 'Mozambique_season_2_all.txt.ori')

# %% codecell
def read_lists(path):

    file_list = []
    label_list = []

    with open(path, 'r') as f:

        for line in tqdm(f):

            line = line.replace('\n', '')
            line_sp = line.split(' ')
            file = line_sp[0]
            label = line_sp[1]

            file_list.append(file)
            label_list.append(label)

    file_list = np.array(file_list)
    label_list = np.array(label_list)

    return file_list, label_list

# %% codecell
def read_ori_lists(path):

    file_list = []
    label_list = []
    sec_list = []

    with open(path, 'r') as f:

        for line in tqdm(f):

            line = line.replace('  ', ' ')

            if 'Honey Badger' in line:
                line = line.replace('Honey Badger', 'Honey_badger')

            if 'Mongoose_white tailed' in line:
                line = line.replace('Mongoose_white tailed', 'Mongoose_white_tailed')

            line_sp = line.split(' ')
            file = line_sp[0]
            label = line_sp[1]
            sec = line_sp[2]

            if file.split('/')[1] in rename.keys():
                file_sp = file.split('/')
                file_sp[1] = rename[file_sp[1]]
                file = '/'.join(file_sp)

            if label in rename.keys():
                label = rename[label]

            file_list.append(file)
            label_list.append(label)
            sec_list.append(float(sec.replace('\n', '')))

    file_list = np.array(file_list)
    label_list = np.array(label_list)
    sec_list = np.array(sec_list)

    return file_list, label_list, sec_list

# %% codecell
def category_selection(label_list, min_count):

    unique_labels, label_counts = np.unique(label_list, return_counts=True)

    cat_sel = [(cat, count)
               for cat, count in zip(unique_labels[label_counts >= min_count], label_counts[label_counts >= min_count])
               if 'nknown' not in cat
               and 'other' not in cat
               and 'Rodent' not in cat
               and 'Mongoose_dwarf' not in cat]

    cat_leftout = [(cat, count)
                   for cat, count in zip(unique_labels, label_counts)
                   if 'nknown' in cat
                   or 'other' in cat
                   or count < min_count
                   or 'Rodent' in cat
                   or 'Mongoose_dwarf' in cat]

    cat_sel = sorted(cat_sel, key=lambda x : x[1], reverse=True)
    cat_leftout = sorted(cat_leftout, key=lambda x : x[1], reverse=True)

    return cat_sel, cat_leftout

# %% markdown
# # 1. Combine Two Seasons And Split

# %% codecell
file_list_s1, label_list_s1, sec_list_s1 = read_ori_lists(s1_ori)
file_list_s2, label_list_s2, sec_list_s2 = read_ori_lists(s2_ori)

# %% codecell
file_list_all = np.concatenate((file_list_s1, file_list_s2), axis=0)
label_list_all = np.concatenate((label_list_s1, label_list_s2), axis=0)
sec_list_all = np.concatenate((sec_list_s1, sec_list_s2), axis=0)

# %% codecell
cat_sel_counts_all, cat_leftout_counts_all = category_selection(label_list_all, min_count=50)

# %% codecell
# len(cat_sel_counts_all)
cat_sel_counts_all

# %%
len(cat_sel_counts_all)

# %% codecell
# len(cat_leftout_counts_all)
len(cat_leftout_counts_all)
cat_leftout_counts_all

# %%
# Distribution Ploting
used_cat = [(cat, c, 0) for cat, c in cat_sel_counts_all]
used_cat += [(cat, c, 1) for cat, c in cat_leftout_counts_all if cat != 'Mongoose_dwarf']
used_cat = sorted(used_cat, key=lambda x:x[1], reverse=True)

cat = [e[0] for e in used_cat]
ct = [e[1] for e in used_cat]
ood = ['Selected as unknown categories' if e[2] else 'Selected as known categories' for e in used_cat]

sns.set(font_scale=2, style='whitegrid')
plt.subplots(figsize=(15, 30))
ax = sns.barplot(x=ct, y=cat, hue=ood, dodge=False)
ax.set_title('Distribution of the whole dataset', fontsize=35, y=1.01)
ax.set_xscale('log')
ax.set_xlabel('Total #', fontsize=25, y=-0.02)
plt.savefig('./Figs/TotalDistribution.png', bbox_inches='tight')

# %% codecell
tr_ood_list = open(os.path.join(root, 'SplitLists', 'train_mix_ood.txt'), 'w')
te_ood_list = open(os.path.join(root, 'SplitLists', 'val_mix_ood.txt'), 'w')
for cat_id, (cat, count) in tqdm(enumerate(cat_leftout_counts_all), total=len(cat_leftout_counts_all)):
    # Select category files, labels, and shooting seconds
    file_sel = file_list_all[label_list_all == cat]
    label_sel = label_list_all[label_list_all == cat]
    sec_sel = sec_list_all[label_list_all == cat]

    if cat_id < 4:
        random.seed(count)

        # Group images by shooting times
        index_group = []
        last_sec = 0.
        same_shoot_index = []
        for index, sec in enumerate(sec_sel):
            if len(same_shoot_index) == 0:
                same_shoot_index.append(index)
            else:
                if (sec - last_sec) < 2:
                    same_shoot_index.append(index)
                else:
                    index_group.append(same_shoot_index)
                    same_shoot_index = [index]
            last_sec = sec
        if len(same_shoot_index) > 0:
            index_group.append(same_shoot_index)

        # Shuffle shooting groups and get random index list
        random.shuffle(index_group)
        index_rand = np.array([i for g in index_group for i in g])

        # Use random indices to shuffle selected files
        file_sel = file_sel[index_rand]

        train_test_split_ratio = 0.2
        test_counts = int(train_test_split_ratio * len(file_sel))

        file_sel_te = file_sel[:test_counts]
        file_sel_tr = file_sel[test_counts:]
        label_sel_te = label_sel[:test_counts]
        label_sel_tr = label_sel[test_counts:]

        for f, l in zip(file_sel_tr, label_sel_tr):
            tr_s2_list.write(f + ' ' + l + '\n')

        for f, l in zip(file_sel_te, label_sel_te):
            te_s2_list.write(f + ' ' + l + '\n')

    else:
        for f, l in zip(file_sel, label_sel):
            te_ood_list.write(f + ' ' + l + '\n')

    for f, l in zip(file_sel, label_sel):
        tr_ood_list.write(f + ' ' + l + '\n')
tr_ood_list.close()
te_ood_list.close()


# %% codecell

tr_s1_list = open(os.path.join(root, 'SplitLists', 'train_mix_season_1_lt.txt'), 'w')
te_s1_list = open(os.path.join(root, 'SplitLists', 'val_mix_season_1_lt.txt'), 'w')
tr_s2_list = open(os.path.join(root, 'SplitLists', 'train_mix_season_2_lt.txt'), 'w')
te_s2_list = open(os.path.join(root, 'SplitLists', 'val_mix_season_2_lt.txt'), 'w')

for cat_id, (cat, count) in tqdm(enumerate(cat_sel_counts_all), total=len(cat_sel_counts_all)):

    random.seed(count)

    # Select category files, labels, and shooting seconds
    file_sel = file_list_all[label_list_all == cat]
    label_sel = label_list_all[label_list_all == cat]
    sec_sel = sec_list_all[label_list_all == cat]

    if cat == 'Ghost':
        ghost_id = np.random.choice(range(len(file_sel)), 50000, replace=False)
        file_sel = file_sel[ghost_id]
        label_sel = label_sel[ghost_id]
        sec_sel = sec_sel[ghost_id]

    # Group images by shooting times
    index_group = []
    last_sec = 0.
    same_shoot_index = []
    for index, sec in enumerate(sec_sel):
        if len(same_shoot_index) == 0:
            same_shoot_index.append(index)
        else:
            if (sec - last_sec) < 2:
                same_shoot_index.append(index)
            else:
                index_group.append(same_shoot_index)
                same_shoot_index = [index]
        last_sec = sec
    if len(same_shoot_index) > 0:
        index_group.append(same_shoot_index)

    # Shuffle shooting groups and get random index list
    random.shuffle(index_group)
    index_rand = np.array([i for g in index_group for i in g])

    # Use random indices to shuffle selected files
    file_sel = file_sel[index_rand]

    # Generate counts for each set
    train_test_split_ratio = 0.2
    test_counts = int(train_test_split_ratio * len(file_sel))

    # For the first 26 classes, generate two_season_split_ratio
    # And save data to corresponding lists
    if cat_id < 26:

        two_season_split_ratio = random.randint(30, 50) / 100
        train_counts_1 = int(two_season_split_ratio * len(file_sel))

        file_sel_te_1 = file_sel[:int(test_counts/2)]
        file_sel_te_2 = file_sel[int(test_counts/2):test_counts]

        file_sel_tr_1 = file_sel[test_counts:(test_counts + train_counts_1)]
        file_sel_tr_2 = file_sel[(test_counts + train_counts_1):]

        label_sel_te_1 = label_sel[:int(test_counts/2)]
        label_sel_te_2 = label_sel[int(test_counts/2):test_counts]

        label_sel_tr_1 = label_sel[test_counts:(test_counts + train_counts_1)]
        label_sel_tr_2 = label_sel[(test_counts + train_counts_1):]

        for f, l in zip(file_sel_tr_1, label_sel_tr_1):
            tr_s1_list.write(f + ' ' + l + '\n')

        for f, l in zip(file_sel_tr_2, label_sel_tr_2):
            tr_s2_list.write(f + ' ' + l + '\n')

        for f, l in zip(file_sel_te_1, label_sel_te_1):
            te_s1_list.write(f + ' ' + l + '\n')

        for f, l in zip(file_sel_te_2, label_sel_te_2):
            te_s2_list.write(f + ' ' + l + '\n')

    else:

        # Only save for second season list
        if test_counts < 50:
            test_counts = 50

        if len(file_sel) < 80:
            test_counts = 20

        file_sel_te_2 = file_sel[:test_counts]
        file_sel_tr_2 = file_sel[test_counts:]
        label_sel_te_2 = label_sel[:test_counts]
        label_sel_tr_2 = label_sel[test_counts:]

        for f, l in zip(file_sel_tr_2, label_sel_tr_2):
            tr_s2_list.write(f + ' ' + l + '\n')

        for f, l in zip(file_sel_te_2, label_sel_te_2):
            te_s2_list.write(f + ' ' + l + '\n')

tr_s1_list.close()
te_s1_list.close()
tr_s2_list.close()
te_s2_list.close()

# %% codecell
paths_tr_s1, labels_tr_s1 = read_lists(os.path.join(root, 'SplitLists', 'train_mix_season_1_lt.txt'))
paths_te_s1, labels_te_s1 = read_lists(os.path.join(root, 'SplitLists', 'val_mix_season_1_lt.txt'))
paths_tr_s2, labels_tr_s2 = read_lists(os.path.join(root, 'SplitLists', 'train_mix_season_2_lt.txt'))
paths_te_s2, labels_te_s2 = read_lists(os.path.join(root, 'SplitLists', 'val_mix_season_2_lt.txt'))

# %%
check_list = []

with open(os.path.join(root, 'SplitLists', 'val_mix_season_2_lt.txt'), 'r') as f:

        for line in tqdm(f):

            line = line.replace('\n', '')
            line_sp = line.split(' ')
            file = line_sp[0]
            label = line_sp[1]

            if label == 'Guineafowl_helmeted':
                check_list.append(file)

# %%
Image.open(os.path.join(root, random.choice(check_list)))

# %% codecell
group_1_list = []

with open(os.path.join(root, 'SplitLists', 'train_mix_season_1_lt.txt'), 'r') as f:
        for line in tqdm(f):
            line = line.replace('\n', '')
            line_sp = line.split(' ')
            label = line_sp[1]
            group_1_list.append(label)

with open(os.path.join(root, 'SplitLists', 'val_mix_season_1_lt.txt'), 'r') as f:
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

with open(os.path.join(root, 'SplitLists', 'val_mix_season_2_lt.txt'), 'r') as f:
        for line in tqdm(f):
            line = line.replace('\n', '')
            line_sp = line.split(' ')
            label = line_sp[1]
            group_2_list.append(label)

# %%
group_1_counts = sorted([(cat, count) 
                         for cat, count in zip(*np.unique(group_1_list,
                                                          return_counts=True))],
                        key=lambda x:x[1], reverse=True)

group_2_counts = sorted([(cat, count) 
                         for cat, count in zip(*np.unique(group_2_list,
                                                          return_counts=True))],
                        key=lambda x:x[1], reverse=True)

# %%
group_counts = pd.DataFrame(columns=['Group'] + [e[0] for e in group_2_counts],
                            data=[['2'] + [e[1] for e in group_2_counts],
                                  ['1'] + [group_1_counts[i][1] if i < len(group_1_counts) - 1 else 0 
                                                 for i in range(len(group_2_counts))]])

# %%
group_counts.set_index('Group').T.plot(kind='barh', stacked=False, figsize=(15, 30))

# group_counts.plot.bar(stacked=True, title="xx")

# %%

# %%
total_counts = [0 for _ in range(len(group_2_counts))]

for i, e in enumerate(group_1_counts):
    total_counts[i] += e[1]

for i, e in enumerate(group_2_counts):
    total_counts[i] += e[1]


# %%
cat = [e[0] for e in group_1_counts]
cat += [e[0] for e in group_2_counts]
group_1 = [e[1] for e in group_1_counts]
group_2 = [e[1] for e in group_2_counts]
ct = group_1 + group_2
hue = ['Group 1' if i < len(group_1) else 'Group 2' for i in range(len(ct))]

sns.set(font_scale=2, style='whitegrid')
plt.subplots(figsize=(15, 30))
ax = sns.barplot(x=ct, y=cat, hue=hue, dodge=True)
ax.set_title('Distribution of Group 1 & 2', fontsize=35, y=1.01)
ax.set_xscale('log')
ax.set_xlabel('Total #', fontsize=25, y=-0.02)
plt.savefig('./Figs/GroupDistribution.png', bbox_inches='tight')
# %%
