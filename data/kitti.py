import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image
from data.transforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor, CenterCrop, RandomRotation, RandomVerticalFlip

resolution_dict = {
    'full' : (384, 1280),
    'tu_small' : (128, 416),
    'tu_big' : (228, 912),
    'half' : (192, 640)}

import os
from PIL import Image

class KITTIDataset(Dataset):
    def __init__(self, root, split, resolution='full', augmentation='alhashim'):

        self.root = root
        self.split = split
        self.resolution = resolution_dict[resolution]
        self.augmentation = augmentation

        self.rgb_dir = os.path.join(self.root, 'data_raw')
        if split == 'train':
            self.transform = self.train_transform
            self.depth_dir = os.path.join(self.root, 'denseDepth', 'train')
        elif split == 'val':
            self.transform = self.val_transform
            self.depth_dir = os.path.join(self.root, 'denseDepth', 'val')
        elif split == 'test':
            if self.augmentation == 'alhashim':
                self.transform = None
            else:
                self.transform = CenterCrop(self.resolution)
            self.depth_dir = os.path.join(self.root, 'denseDepth', 'test') # não tem teste

        self.image_pairs = self.get_image_pairs()

    def get_image_pairs(self):
        mode_depth_path = self.depth_dir
        image_pairs = []

        for root, _, files in os.walk(mode_depth_path):
            for file in files:
                if file.endswith('.png'):
                    depth_img_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, mode_depth_path)
                    drive_id = relative_path.split(os.path.sep)[0]

                    rgb_drive_path = os.path.join(self.rgb_dir, drive_id, drive_id.split('_drive')[0], drive_id, 'image_02', 'data')
                    rgb_img_path = os.path.join(rgb_drive_path, file)

                    if os.path.isfile(rgb_img_path):
                        image_pairs.append((depth_img_path, rgb_img_path))

        return image_pairs

    def __getitem__(self, index):
        depth_path, rgb_path = self.image_pairs[index]
        depth = Image.open(depth_path)
        image = Image.open(rgb_path)

        data = {'depth': depth, 'image': image}

        if self.transform is not None:
            data = self.transform(data)

        image, depth = data['image'], data['depth']
        if self.split == 'test':
            image = np.array(image)
            depth = np.array(depth)
        return image, depth

    def __len__(self):
        return len(self.image_pairs)


    def train_transform(self, data):
        if self.augmentation == 'alhashim':
            transform = Compose([
                Resize(self.resolution),
                RandomHorizontalFlip(),
                RandomChannelSwap(0.25),
                ToTensor(test=False, maxDepth=80.0)
            ])
        else:
            transform = Compose([
                RandomRotation(4.5),
                CenterCrop(self.resolution),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomChannelSwap(0.25),
                ToTensor(test=False, maxDepth=80.0)
            ])

        data = transform(data)
        return data

    def val_transform(self, data):
        if self.augmentation == 'alhashim':
            transform = Compose([
                Resize(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])
        else:
            transform = Compose([
                CenterCrop(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])

        data = transform(data)
        return data
