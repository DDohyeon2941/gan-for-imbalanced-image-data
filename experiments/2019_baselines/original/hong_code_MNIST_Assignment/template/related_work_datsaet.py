# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:06:53 2019

@author: User
"""

# import some packages you need here
from __future__ import print_function, division
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
import re
from imgaug import augmenters as iaa
import time


class MNIST(Dataset):
    """ MNIST dataset
        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    Args:
        data_dir: directory path containing images
    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, aug_option= False):
        
        self.data_dir= data_dir
        self.aug_option= aug_option
        
        self.data_list= glob('%s/*.png'%data_dir)
        
        if self.aug_option:
            self.data_list.extend(self.data_list)
            
        self.transform= transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean= [0.1307],
                                     std= [0.3081])
            ])
        

    def __len__(self):
        
        return len(self.data_list)
    

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx= idx.tolist()
            
        img_name= self.data_list[idx]
        img= Image.open(img_name)
                    
        if idx>= len(self.data_list):
            
            img= self.transform(self.aug(img))
            
        else:
            
            img= self.transform(img)
            
        label= int(re.findall(r'\d+_\d', img_name)[0].split('_')[1])
        
        return img, label
    
    
    def aug(self, img):
        
        seq= iaa.Sequential([
                iaa.GammaContrast(gamma= 1.5),
                iaa.GaussianBlur(sigma= (0, 2.))])
        
        return seq(images= img)
    

if __name__ == '__main__':

    train_dir= '../data/train'
    batch_size= 256

    before= time.time()

    train_dataset= MNIST(train_dir)
    
    train_loader= DataLoader(dataset=train_dataset, 
                             batch_size=batch_size,
                             shuffle=True)
    
    for batch_idx, (x, target) in enumerate(train_loader):
        if batch_idx% 10== 0:
            print(x.shape, target.size())
            print(len(train_loader.dataset))
            
    print(time.time()- before)
    
    
    before= time.time()
            
    train_dataset_aug= MNIST(train_dir, aug_option= True)
    
    train_loader_aug= DataLoader(dataset= train_dataset_aug,
                                 batch_size= batch_size,
                                 shuffle= True)
    
    for batch_idx, (x, target) in enumerate(train_loader_aug):
        if batch_idx% 10== 0:
            print(x.shape, target.size())
            print(len(train_loader_aug.dataset))
            
    print(time.time()- before)