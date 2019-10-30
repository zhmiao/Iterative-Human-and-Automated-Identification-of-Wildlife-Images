import os
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
import paramiko


def upload(season):

    root = os.path.join('/Volumes/New Volume', season)
    info_txt = open('./{}.txt'.format(season), 'a')

    srv_root = '/home/zhmiao/datasets/ecology/Mozambique/'

    server = 'localhost'
    username = 'zhmiao'
    password = 'Onepiece1%'

    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))

    ssh.connect(hostname=server, username=username, password=password, port=2222)
    sftp = ssh.open_sftp()

    cam_list = [d for d in os.listdir(root) if not d.startswith('.') and os.path.isdir(os.path.join(root, d))]

    for cam in cam_list:

        cam_root = os.path.join(root, cam)

        species_list = [d for d in os.listdir(cam_root) if
                        not d.startswith('.') and os.path.isdir(os.path.join(cam_root, d))]

        for species in species_list:

            species_root = os.path.join(cam_root, species)

            file_list = [d for d in os.listdir(species_root) if not d.startswith('.') and d.endswith('JPG')]

            for local_file_name in file_list:

                img_path = os.path.join(species_root, local_file_name)

                srv_species_root = os.path.join(srv_root, season, species)

                try:
                    sftp.stat(srv_species_root)
                except IOError:
                    print('Create {}'.format(srv_species_root))
                    ssh.exec_command('mkdir -p {}'.format(srv_species_root))

                srv_file_name = cam + '_' + local_file_name.replace(' ', '_')
                srv_img_path = os.path.join(srv_species_root, srv_file_name)
                file_id = srv_img_path.replace(srv_root, '')
                assert ' ' not in file_id

                fl = BytesIO()

                try:
                    img = Image.open(img_path)
                except:
                    print('Image loading problem')
                    print(img_path)

                try:
                    time_stamp = img._getexif()[36867]
                    time_stamp = datetime.strptime(time_stamp, '%Y:%m:%d %H:%M:%S').timestamp()
                except:
                    print('Datetime loading problem')
                    print(img_path)

                w, h = img.size
                img = img.crop((0, 0, w, int(h * 0.93))).resize((256, 256))
                img.save(fl, format='JPEG')
                file_size = fl.tell()

                fl.seek(0)

                sftp.putfo(fl, srv_img_path, file_size, None, True)

                fl.close()

                info_txt.write('{} {} {}\n'.format(file_id, species, time_stamp))

    sftp.close()
    ssh.close()

    info_txt.close()


for season in ['Mozambique_season_1', 'Mozambique_season_2']:
    upload(season)
