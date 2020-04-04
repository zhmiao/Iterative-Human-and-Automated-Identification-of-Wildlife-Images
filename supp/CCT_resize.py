import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

ori_root = '/home/zhmiao/datasets/ecology/CCT_15/eccv_18_all_images_sm'


def resize(img_file):
    img_obj = Image.open(os.path.join(ori_root, img_file))
    img_obj.resize((256, 256))
    img_obj.save(os.path.join(ori_root.replace('_sm', '_256'), img_file), format='JPEG')


img_file_list = os.listdir(ori_root)
pool = Pool(processes=20)
for _ in tqdm(pool.imap_unordered(resize, img_file_list, chunksize=20), total=len(img_file_list)):
    pass
pool.close()

