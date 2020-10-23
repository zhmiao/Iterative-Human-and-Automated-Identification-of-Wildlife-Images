import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.data.class_aware_sampler import ClassAwareSampler

# Standard data transform with resize and typical augmentation
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(3.5/4.0, 3.5/3.0)),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.5/4.0, 3.5/3.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_strong': transforms.Compose([
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.5/4.0, 3.5/3.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45, fill=(123, 116, 103)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'MOZ': transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'eval': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


dataset_obj = {}
def register_dataset_obj(name):

    """
    Dataset register
    """

    def decorator(cls):
        dataset_obj[name] = cls
        return cls
    return decorator


def get_dataset(name, rootdir, class_indices, dset, transform, **add_args):

    """
    Dataset getter
    """

    print('Getting dataset: {} {} {} \n'.format(name, rootdir, dset))

    return dataset_obj[name](rootdir, class_indices=class_indices, dset=dset, 
                             transform=data_transforms[transform], **add_args)


def load_dataset(name, class_indices, dset, transform, batch_size=64, rootdir='',
                 shuffle=True, num_workers=1, cas_sampler=False, **add_args):

    """
    Dataset loader
    """

    if dset != 'train':
        shuffle = False

    print('Shuffle is {}.'.format(shuffle))

    dataset = get_dataset(name, rootdir, class_indices, dset, transform, **add_args)

    if len(dataset) == 0:
        return None

    if cas_sampler:
        print("** USING CAS SAMPLER!! **")
        # TODO, sampler numbers
        sampler = ClassAwareSampler(dataset.labels, 3)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return loader


class BaseDataset(Dataset):

    def __init__(self, class_indices, dset='train', transform=None):
        self.img_root = None
        self.ann_root = None
        self.class_indices = class_indices
        self.dset = dset
        self.transform = transform
        self.data = []
        self.labels = []

    def load_data(self, ann_dir):
        pass

    def class_counts_cal(self):
        unique_labels, unique_counts = np.unique(self.labels, return_counts=True)
        return unique_labels, unique_counts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        file_id = self.data[index]
        label = self.labels[index]
        file_dir = os.path.join(self.img_root, file_id)
        if not file_dir.endswith('.JPG'):
            file_dir += '.jpg'

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


