# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:16:41 2019

@author: User
"""

import torch
from model import netD,netG
#from main import get_sample_image
from matplotlib.pyplot import imshow, imsave
import os
from PIL import Image
import numpy as np
from torchvision.utils import save_image

def get_sample_single_image(G, counts, n_noise, DEVICE, file_path, save):
    #torch.radn(a,b) => b 크기의 데이터 a개를 만들어라
    z = torch.randn(counts, n_noise, 1, 1).to(DEVICE)
    y_hat = torch.squeeze(G(z), 1) # (100, 28, 28)
    

    if save == True:
        for index in range(y_hat.shape[0]):
            new_image=y_hat.data[index]
            save_image(new_image,file_path+'/%s_%s.png'%(index,1))
            print(index)

    return ''

if __name__ == '__main__':

    
    path=r'./models/AC_GAN/cifar10/G_99_epoch.pkl'

    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=64
    n_noise=100 
    
    nz = 100
    ngf = 32
    #ndf = 32
    nc = 3
    
    model=torch.nn.DataParallel(netG(nz, ngf, nc),device_ids=[0])
    model.load_state_dict(torch.load(path))
    model.eval()
    
    file_path=r'./oversampled_data'    
    if not os.path.exists(file_path):   
        os.makedirs(file_path)

    
    counts=100
    get_sample_single_image(model, counts, n_noise,DEVICE,file_path,True)
    
'''
    if save == True:
        for index in range(result.shape[0]):
            imsave(file_path+'\%s_%s.png'%(index, index), Image.fromarray((result[index]*255).astype(int).reshape(28,28)), cmap='gray')

'''
