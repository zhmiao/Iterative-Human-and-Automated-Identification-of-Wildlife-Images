# %% codecell
import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image

# %% codecell
rename = {
    'Baboon_': 'Baboon',
    'Elephant_': 'Elephant',
    'Honey_badger_': 'Honey_badger',
    'Honey_Badger': 'Honey_badger',
    'Mongoose_larger_gray': 'Mongoose_larger_grey',
    'Hippopotamus': 'Hippo'
}

# %% codecell
root = '/home/zhmiao/datasets/ecology/Mozambique/'
s1_ori = os.path.join(root, 'SplitLists', 'Mozambique_season_1_all.txt.ori')
s2_ori = os.path.join(root, 'SplitLists', 'Mozambique_season_2_all.txt.ori')

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

            file_list.append(file)
            label_list.append(label)
            sec_list.append(float(sec.replace('\n', '')))

    file_list = np.array(file_list)
    label_list = np.array(label_list)
    sec_list = np.array(sec_list)

    return file_list, label_list, sec_list

# %% codecell
file_list_s1, label_list_s1, sec_list_s1 = read_ori_lists(s1_ori)
file_list_s2, label_list_s2, sec_list_s2 = read_ori_lists(s2_ori)

# %% codecell
def category_selection(label_list, min_count):

    unique_labels, label_counts = np.unique(label_list, return_counts=True)

    cat_sel = [(cat, count)
               for cat, count in zip(unique_labels[label_counts > min_count], label_counts[label_counts > min_count])
               if 'nknown' not in cat
               and 'other' not in cat
               and cat not in ['Ghost', 'Human', 'Fire', 'Setup', 'Rodent']]

    cat_leftout = [(cat, count)
                   for cat, count in zip(unique_labels, label_counts)
                   if 'nknown' in cat
                   or 'other' in cat
                   or cat in ['Ghost', 'Human', 'Fire', 'Setup', 'Rodent']
                   or count <= min_count]

    cat_sel = sorted(cat_sel, key=lambda x : x[1], reverse=True)
    cat_leftout = sorted(cat_leftout, key=lambda x : x[1], reverse=True)

    return cat_sel, cat_leftout

# %% codecell
cat_sel_counts_s1, cat_leftout_counts_s1 = category_selection(label_list_s1, min_count=150)
cat_sel_counts_s2, cat_leftout_counts_s2 = category_selection(label_list_s2, min_count=150)

# %% markdown
# # 1. Combine Two Seasons

# %% codecell
file_list_all = np.concatenate((file_list_s1, file_list_s2), axis=0)
label_list_all = np.concatenate((label_list_s1, label_list_s2), axis=0)
sec_list_all = np.concatenate((sec_list_s1, sec_list_s2), axis=0)

# %% codecell
cat_sel_counts_all, cat_leftout_counts_all = category_selection(label_list_all, min_count=150)

# %% codecell

tr_s1_list = open(os.path.join(root, 'SplitLists', 'train_mix_season_1.txt'), 'w')
te_s1_list = open(os.path.join(root, 'SplitLists', 'test_mix_season_1.txt'), 'w')
tr_s2_list = open(os.path.join(root, 'SplitLists', 'train_mix_season_2.txt'), 'w')
te_s2_list = open(os.path.join(root, 'SplitLists', 'test_mix_season_2.txt'), 'w')

for cat_id, (cat, count) in tqdm(enumerate(cat_sel_counts_all), total=len(cat_sel_counts_all)):

    random.seed(count)

    # Select category files, labels, and shooting seconds
    file_sel = file_list_all[label_list_all == cat]
    label_sel = label_list_all[label_list_all == cat]
    sec_sel = sec_list_all[label_list_all == cat]

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

    # For the first 20 classes, generate two_season_split_ratio
    # And save data to corresponding lists
    if cat_id < 20:

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

# %% markdown
# # 2. Two season class indices

# %% codecell
# Load lists first
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
root = '/home/zhmiao/datasets/ecology/Mozambique/'
tr_s1 = os.path.join(root, 'SplitLists', 'train_mix_season_1.txt')
tr_s2 = os.path.join(root, 'SplitLists', 'train_mix_season_2.txt')
te_s1 = os.path.join(root, 'SplitLists', 'test_mix_season_1.txt')
te_s2 = os.path.join(root, 'SplitLists', 'test_mix_season_2.txt')

# %% codecell
file_list_tr_s1, label_list_tr_s1 = read_lists(tr_s1)
file_list_te_s1, label_list_te_s1 = read_lists(te_s1)

file_list_tr_s2, label_list_tr_s2 = read_lists(tr_s2)
file_list_te_s2, label_list_te_s2 = read_lists(te_s2)

# %% codecell
sorted_cat_s1 = sorted(list(zip(*np.unique(label_list_tr_s1, return_counts=True))), key=lambda x:x[1], reverse=True)
sorted_cat_s2 = sorted(list(zip(*np.unique(label_list_tr_s2, return_counts=True))), key=lambda x:x[1], reverse=True)

# %% codecell
class_indices_s1 = {e[0]:i for i, e in enumerate(sorted_cat_s1)}

# %% codecell
class_indices_s1

# %% codecell
add_class_indices_s2 = {}

i = 0
for e in sorted_cat_s2:
    if e[0] not in class_indices_s1:
        add_class_indices_s2[e[0]] = i + 20
        i += 1

# %% codecell
add_class_indices_s2

# %% markdown
# # 3. Empty picker datasets

# %% codecell
np.unique(label_list_all)
empty_files = file_list_all[label_list_all == 'Ghost']

non_empty_files = file_list_all[(label_list_all != 'Ghost')\
                                & (label_list_all != 'Human')\
                                & (label_list_all != 'Setup')\
                                & (label_list_all != 'Fire')]

non_empty_labels = label_list_all[(label_list_all != 'Ghost')\
                                 & (label_list_all != 'Human')\
                                 & (label_list_all != 'Setup')\
                                 & (label_list_all != 'Fire')]

non_empty_sec = sec_list_all[(label_list_all != 'Ghost')\
                             & (label_list_all != 'Human')\
                             & (label_list_all != 'Setup')\
                             & (label_list_all != 'Fire')]

non_empty_label_counts = sorted(list(zip(*np.unique(non_empty_labels, return_counts=True))), key=lambda x:x[1])

non_empty_label_counts = [e for e in non_empty_label_counts if e[1] > 10]

# %% codecell
non_empty_label_counts

# %% codecell

empty_tr_list = open(os.path.join(root, 'SplitLists', 'train_empty.txt'), 'w')
empty_te_list = open(os.path.join(root, 'SplitLists', 'test_empty.txt'), 'w')

test_ratio = 0.1

# %% codecell
# Select empty files first
np.random.seed(len(empty_files))
np.random.shuffle(empty_files)

test_empty_counts = int(len(empty_files) * test_ratio)

file_empty_te = empty_files[:test_empty_counts]
file_empty_tr = empty_files[test_empty_counts:]
empty_labels_te = np.zeros(len(file_empty_te)).astype(np.int64)
empty_labels_tr = np.zeros(len(file_empty_tr)).astype(np.int64)

for f, l in zip(file_empty_tr, empty_labels_tr):
    empty_tr_list.write(f + ' ' + str(l) + '\n')
for f, l in zip(file_empty_te, empty_labels_te):
    empty_te_list.write(f + ' ' + str(l) + '\n')

# %% codecell
for cat_id, (cat, count) in tqdm(enumerate(non_empty_label_counts), total=len(non_empty_label_counts)):

    random.seed(count)

    # Select category files, labels, and shooting seconds
    file_sel = non_empty_files[non_empty_labels == cat]
    label_sel = non_empty_labels[non_empty_labels == cat]
    sec_sel = non_empty_sec[non_empty_labels == cat]

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

    # Generate counts for each set
    train_test_split_ratio = 0.1
    test_counts = int(train_test_split_ratio * len(file_sel))

    # Only save for second season list
    if test_counts < 50:
        test_counts = 50

    file_sel_te = file_sel[:test_counts]
    file_sel_tr = file_sel[test_counts:]
    label_sel_te = np.ones(len(label_sel[:test_counts])).astype(np.int64)
    label_sel_tr = np.ones(len(label_sel[test_counts:])).astype(np.int64)

    for f, l in zip(file_sel_tr, label_sel_tr):
        empty_tr_list.write(f + ' ' + str(l) + '\n')
    for f, l in zip(file_sel_te, label_sel_te):
        empty_te_list.write(f + ' ' + str(l) + '\n')

empty_tr_list.close()
empty_te_list.close()

# %% markdown
# # 4. Additional Datasets

# %% codecell
tr_10_s1_list = open(os.path.join(root, 'SplitLists', 'train_ori_10_season_1.txt'), 'w')
te_10_s1_list = open(os.path.join(root, 'SplitLists', 'test_ori_10_season_1.txt'), 'w')

############
# cat_id = 0
# cat = cat_sel_counts_s1[0][0]
# count = cat_sel_counts_s1[0][1]
############

for cat_id, (cat, count) in tqdm(enumerate(cat_sel_counts_s1), total=len(cat_sel_counts_s1)):

    random.seed(count)

    # Select category files, labels, and shooting seconds
    file_sel = file_list_s1[label_list_s1 == cat]
    label_sel = label_list_s1[label_list_s1 == cat]
    # label_sel = np.array(cat_id for _ in range(len(file_sel)))
    sec_sel = sec_list_s1[label_list_s1 == cat]

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

    # TODO: Implement index_rand!!!!

    # Use random indices to shuffle selected files
    file_sel = file_sel[index_rand]

    # Generate counts for each set
    train_test_split_ratio = 0.1
    test_counts = int(train_test_split_ratio * len(file_sel))

    # For the first 20 classes, generate two_season_split_ratio
    # And save data to corresponding lists
    if cat_id < 10:

        file_sel_te_1 = file_sel[:test_counts]
        file_sel_tr_1 = file_sel[test_counts:]

        label_sel_te_1 = label_sel[:test_counts]
        label_sel_tr_1 = label_sel[test_counts:]

        for f, l in zip(file_sel_tr_1, label_sel_tr_1):
            tr_10_s1_list.write(f + ' ' + l + '\n')

        for f, l in zip(file_sel_te_1, label_sel_te_1):
            te_10_s1_list.write(f + ' ' + l + '\n')

tr_10_s1_list.close()
te_10_s1_list.close()

# %% codecell
tr_10_s1_file, tr_10_s1_labels = read_lists(os.path.join(root, 'SplitLists', 'train_ori_10_season_1.txt'))
{e[0]: i for i, e in enumerate(sorted(list(zip(*np.unique(tr_10_s1_labels, return_counts=True))), key=lambda x:x[1], reverse=True))}


# %% codecell
tr_10_mix_list = open(os.path.join(root, 'SplitLists', 'train_mix_10_season_1.txt'), 'w')
te_10_mix_list = open(os.path.join(root, 'SplitLists', 'test_mix_10_season_1.txt'), 'w')

############
# cat_id = 0
# cat = cat_sel_counts_all[0][0]
# count = cat_sel_counts_all[0][1]
############

for cat_id, (cat, count) in tqdm(enumerate(cat_sel_counts_all), total=len(cat_sel_counts_all)):

    random.seed(count)

    # Select category files, labels, and shooting seconds
    file_sel = file_list_all[label_list_all == cat]
    label_sel = label_list_all[label_list_all == cat]
    sec_sel = sec_list_all[label_list_all == cat]

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

    # TODO: Implement index_rand!!!!

    # Use random indices to shuffle selected files
    file_sel = file_sel[index_rand]

    # Generate counts for each set
    train_test_split_ratio = 0.1
    test_counts = int(train_test_split_ratio * len(file_sel))

    # For the first 20 classes, generate two_season_split_ratio
    # And save data to corresponding lists
    if cat_id < 10:

        two_season_split_ratio = random.randint(30, 50) / 100
        train_counts_1 = int(two_season_split_ratio * len(file_sel))

        file_sel_te_1 = file_sel[:int(test_counts/2)]
        file_sel_tr_1 = file_sel[test_counts:(test_counts + train_counts_1)]

        label_sel_te_1 = label_sel[:int(test_counts/2)]
        label_sel_tr_1 = label_sel[test_counts:(test_counts + train_counts_1)]

        for f, l in zip(file_sel_tr_1, label_sel_tr_1):
            tr_10_mix_list.write(f + ' ' + l + '\n')

        for f, l in zip(file_sel_te_1, label_sel_te_1):
            te_10_mix_list.write(f + ' ' + l + '\n')

tr_10_mix_list.close()
te_10_mix_list.close()

# %% codecell
tr_10_mix_file, tr_10_mix_labels = read_lists(os.path.join(root, 'SplitLists', 'train_mix_10_season_1.txt'))
{e[0]: i for i, e in enumerate(sorted(list(zip(*np.unique(tr_10_mix_labels, return_counts=True))), key=lambda x:x[1], reverse=True))}
