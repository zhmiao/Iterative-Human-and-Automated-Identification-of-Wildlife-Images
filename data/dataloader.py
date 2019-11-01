import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Standard data transform with resize and typical augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Dataset dictionary
dataset_obj = {}


def register_dataset_obj(name):

    """
    Dataset register
    """

    def decorator(cls):
        dataset_obj[name] = cls
        return cls
    return decorator


def get_dataset(name, rootdir, dset):

    """
    Dataset getter
    """

    print('Getting dataset: {} {} {} \n'.format(name, rootdir, dset))

    return dataset_obj[name](rootdir, dset=dset, transform=data_transforms)


def load_dataset(name, dset, batch=64, rootdir='', shuffle=True, num_workers=1):

    """
    Dataset loader
    """

    if dset != 'train':
        shuffle = False

    dataset = get_dataset(name, rootdir, dset)

    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return loader


