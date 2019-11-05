import os
from PIL import Image
from tqdm import tqdm

ori_root = '/home/zhmiao/datasets/ecology/CCT_15/eccv_18_all_images_sm'

for img in tqdm(os.listdir(ori_root)):
    img_obj = Image.open(os.path.join(ori_root, img))
    img_obj.resize((256, 256))
    img_obj.save(os.path.join(ori_root.replace('_sm', '_256'), img), format='JPEG')

