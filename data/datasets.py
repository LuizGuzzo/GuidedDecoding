import os

from torch.utils.data import DataLoader

from data.kitti import KITTIDataset
from data.nyu_reduced import get_NYU_dataset
from data.diode import get_diode_dataset

"""
Preparation of dataloaders for Datasets
"""

def get_dataloader(dataset_name, 
                   path,
                   split='train', 
                   resolution='full',
                   augmentation='alhashim',
                   interpolation='linear', 
                   batch_size=8,
                   workers=4, 
                   uncompressed=False):
    if dataset_name == 'kitti':
        dataset = KITTIDataset(path, 
                split, 
                resolution=resolution)
    elif dataset_name == 'nyu_reduced':
        dataset = get_NYU_dataset(path, 
                split, 
                resolution=resolution, 
                uncompressed=uncompressed)
    elif dataset_name == "diode":
        dataset = get_diode_dataset(path, 
                split, 
                resolution=resolution)
    else:
        print('Dataset not existant')
        exit(0)

    dataloader = DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=(split=='train'),
            num_workers=workers, 
            pin_memory=True)
    return dataloader
