import pandas as pd
import numpy as np
import torch
import os
import tarfile
   
# from zipfile import ZipFile
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from data.transforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor

resolution_dict = {
    'full' : (480, 640),
    'half' : (240, 320),
    'mini' : (224, 224)}

class depthDatasetMemory(Dataset): # 4
    def __init__(self, data, split, diode_train, transform=None):
        self.data, self.diode_dataset = data, diode_train
        self.transform = transform
        self.split = split

    def __getitem__(self, idx):
        sample = self.diode_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = np.load(BytesIO(self.data[sample[1]])).squeeze()

        image = np.array(image).astype(np.float32)
        depth = np.array(depth).astype(np.float32)

        # min depth é 0.1 do diode
        if self.split == 'train':
            depth = depth /255.0 * 10.0 #From 8bit to range [0, 10] (meter)
        elif self.split == 'val':
            depth = depth * 0.001
        elif self.split == 'test': # mesma coisa do val
            depth = depth /1000

        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.diode_dataset)



def loadZipToMem(zip_file): # 2

    # Load zip file into memory
    print('Loading dataset zip file...', end='')

    with tarfile.open(zip_file, 'r:gz') as tar:
        data = {}
        for member in tar.getmembers():
            if member.isfile() and (member.name.endswith('.png') or member.name.endswith('.npy')):
                f = tar.extractfile(member)
                if member.name.endswith('.npy'):
                    conteudo = f.read()
                    # array_file = BytesIO(f.read())
                    # array_file.seek(0)
                    # conteudo = np.load(array_file).squeeze()
                    # conteudo = Image.fromarray(conteudo)
                else:
                    conteudo = f.read()
                    # conteudo = Image.open(f)
                data[member.name] = conteudo

    

    # image, depth, mask = Triples
    triples = []
    for fullfilename in data.keys():
        if fullfilename.endswith('.png'):
            filename = fullfilename.split('.')[0]
            triples.append([
                filename + '.png',
                filename + '_depth.npy',
                filename + '_depth_mask.npy'
            ])
        

    # # Debugging
    # if True: diode_train = diode_train[:100]
    # if True: diode_test = diode_test[:100]

    # print('Loaded (Train Images: {0}, Test Images: {1}).'.format(len(diode_train), len(diode_test)))
    print('Loaded (Images: {0}, Triples: {1}).'.format(len(data.keys()), len(triples)))
    return data, triples


def train_transform(resolution): # 3
    transform = transforms.Compose([
        Resize(resolution),
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor(test=False, maxDepth=10.0)
    ])
    return transform

def val_transform(resolution):
    transform = transforms.Compose([
        Resize(resolution),
        ToTensor(test=True, maxDepth=10.0)
    ])
    return transform


def get_diode_dataset(zip_path, split, resolution='full', uncompressed=False): # 1
    resolution = resolution_dict[resolution]
    print("[",split,"] - ",end="")
    if split == 'train':
        data, diode_train = loadZipToMem(zip_path)

        transform = train_transform(resolution)
        dataset = depthDatasetMemory(data, split, diode_train, transform=transform)
    elif split == 'val':
        data, diode_train, diode_test = loadZipToMem(zip_path)

        transform = val_transform(resolution)
        dataset = depthDatasetMemory(data, split, diode_test, transform=transform)
    elif split == 'test':
        # if uncompressed:
        #     dataset = diode_Testset_Extracted(zip_path)
        # else:
        #     dataset = diode_Testset(zip_path)
        data, diode_train, diode_test = loadZipToMem(zip_path)

        transform = val_transform(resolution)
        dataset = depthDatasetMemory(data, split, diode_test, transform=transform)

    return dataset
