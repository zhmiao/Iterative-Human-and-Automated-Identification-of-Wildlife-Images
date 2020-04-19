# %% codecell
import os
from glob import glob
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

# %% codecell
root = '/home/zhmiao/datasets/ecology'
raw_root = os.path.join(root, 'GNP')
save_root = os.path.join(root, 'Mozambique/Mozambique_season_3')

# %% markdown
# # 1. Save raw list

# %% codecell
raw_images = glob(os.path.join(raw_root, '**/*.JPG'), recursive=True)

# %% codecell
raw_list = open(os.path.join(raw_root, 'all_files.txt'), 'w')

for f in tqdm(raw_images):
    raw_list.write(f.replace(raw_root + '/', '') + '\n')

raw_list.close()


# %% markdown
# # 2. Resize and save

# %% codecell
raw_list = open(os.path.join(raw_root, 'all_files.txt'), 'r')

all_files = []

for f in tqdm(raw_list):
    f = f.replace('\n', '')
    all_files.append(f)

raw_list.close()


# %% codecell

def img_resize(f):
    save_dir = os.path.join(save_root, f)
    if not os.path.exists(save_dir):
        try:
            img = Image.open(os.path.join(raw_root, f))
            # Resize
            w, h = img.size
            img = img.crop((0, 0, w, int(h * 0.93))).resize((256, 256))
            # Save
            os.makedirs(save_dir.rsplit('/', 1)[0], exist_ok=True)
            img.save(save_dir)
        except:
            pass
