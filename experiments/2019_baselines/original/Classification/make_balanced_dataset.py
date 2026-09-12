# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:13:47 2019

@author: User
"""

from torch.utils.data import Dataset, DataLoader,ConcatDataset
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets
from PIL import Image
import dataset
from torch.utils import data
import multiprocessing

n_cpu= multiprocessing.cpu_count()


train_path1=r'./data/untar/train/train/'
train_path2=r'./oversampled_data/'

Train_dataset_A = dataset.MNIST(data_dir=train_path1)
Train_dataset_B = dataset.MNIST(data_dir=train_path2)

concat_dataset = ConcatDataset((Train_dataset_A, Train_dataset_B))

aaa=DataLoader(Train_dataset_A, batch_size=64, shuffle=True)
bbb=DataLoader(Train_dataset_B, batch_size=64, shuffle=True)
Train_loader= DataLoader(concat_dataset, batch_size=64, shuffle=True,  num_workers=n_cpu, drop_last=True)
#%%
for batch_idx, (x,target) in enumerate(Train_loader):
    if batch_idx % 50 ==0:
        print(target[:10])
        #print(x[:10])
        
