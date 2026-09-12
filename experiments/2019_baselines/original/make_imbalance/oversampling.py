# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:16:41 2019

@author: User
"""

import torch
from model import Discriminator,Generator
from matplotlib.pyplot import imshow, imsave
import os
from PIL import Image
import numpy as np

def get_sample_image(G, n_noise, DEVICE, file_path,save):
    """
        save sample 100 images
    """
    z = torch.randn(100, n_noise).to(DEVICE)
 
    y_hat = G(z).view(100, 28, 28) # (100, 28, 28)

    result = y_hat.cpu().data.numpy()

    img = np.zeros([280, 280])
    if save == True:
        for j in range(10):
            img[j*28:(j+1)*28] = np.concatenate([x for x in result[j*10:(j+1)*10]], axis=-1)
        new_image=Image.fromarray(img)
        new_image=new_image.convert(mode='1')
        imsave(file_path+'\%s_epoch_result.png'%(2), new_image, cmap='gray')


    return img


def get_sample_single_image(G, counts ,n_noise,DEVICE,file_path,save):
    #torch.radn(a,b) => b 크기의 데이터 a개를 만들어라
    z = torch.randn(counts, n_noise).to(DEVICE)
    
    y_hat = G(z).view(counts, 28, 28) # (100, 28, 28)
#    print(y_hat)
    result = (y_hat.cpu().data.numpy()*255).astype(int)
    print(result.shape)
    print(result.shape[0])
#    print(result)

            
            
    if save == True:
        for index in range(result.shape[0]):
            new_image=Image.fromarray(result[index])
            new_image=new_image.convert(mode='1')
            imsave(file_path+'\%s_%s.png'%(index,1), new_image, cmap='gray')
#            print(result[index][:2])
#            print(result[index].shape)
            print(index)
        
    return ''

if __name__ == '__main__':


    path=r'.\models\5_number\30_epoch\G.pkl'
    DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size=64
    n_noise=100 
    
    model=Generator(n_noise)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    if not os.path.exists(r'.\oversampled_training_data'):   
        os.makedirs(r'.\oversampled_training_data')

    file_path=r'.\oversampled_training_data'
    
    counts=60000
    get_sample_single_image(model, counts, n_noise,DEVICE,file_path,False)
    
    get_sample_image(model, n_noise, DEVICE, file_path,True)
    
#%%
    '''
    if save == True:
        for index in range(result.shape[0]):
            imsave(file_path+'\%s_%s.png'%(index, index), Image.fromarray((result[index]*255).astype(int).reshape(28,28)), cmap='gray')
'''

