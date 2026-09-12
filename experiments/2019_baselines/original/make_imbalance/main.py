# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 01:05:36 2019

@author: User
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from model import Discriminator,Generator
import time
import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib.pyplot import imshow, imsave
import pickle
from make_data_imbalanced import imbalanced_data
import torchvision.datasets




def main():
    
    test_seq = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(
            10, shear=10
        )] + test_seq)
    
    train_dataset = torchvision.datasets.MNIST('./download_data', train=True, download=True, transform=train_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu,drop_last=True)
    DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ##원래 이자리


#    max_epoch = 30
    step=0
    n_critic=1
    D_labels = torch.ones(batch_size, 1).to(DEVICE) # Discriminator Label to real
    D_fakes = torch.zeros(batch_size, 1).to(DEVICE) # Discriminator Label to fake
    
    c_time=0
    
    for number in np.arange(10):
        
        imbalanced_dataset,imbalanced_data_loader=imbalanced_data(train_dataset, train_loader, number, 0.8, n_cpu, batch_size)
        
        
        for candi_epoch in [30,40,50]:
            
            
            
            D = Discriminator().to(DEVICE)
            G = Generator(n_noise).to(DEVICE)

            criterion = nn.BCELoss()
            D_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
            G_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
            step=0
            for epoch in np.arange(candi_epoch):
                
                start=time.time()
        
                for idx, (images, labels) in enumerate(imbalanced_data_loader):
                    # Training Discriminator
                    x = images.to(DEVICE)
                    x_outputs = D(x)
                    
        #            print('-'*50)
        #            print('%s th learing, %s :'% (idx,D_labels.shape))
        #            print('*'*50)
        #            print('%s th learning, %s :'% (idx,x_outputs.shape))
        #            print('-'*50)
                    D_x_loss = criterion(x_outputs, D_labels)
            
                    z = torch.randn(batch_size, n_noise).to(DEVICE)
                    z_outputs = D(G(z))
                    D_z_loss = criterion(z_outputs, D_fakes)
                    
                    D_loss = D_x_loss + D_z_loss
                    
                    D.zero_grad()
                    D_loss.backward()
                    D_opt.step()
            
                    if step % n_critic == 0:
                        # Training Generator
                        z = torch.randn(batch_size, n_noise).to(DEVICE)
                        z_outputs = D(G(z))
                        G_loss = criterion(z_outputs, D_labels)
            
                        G.zero_grad()
                        G_loss.backward()
                        G_opt.step()
                    
                    if step % 500 == 0:
                        c_time=c_time+(time.time() - start)
                        print('Epoch: %s/%s, Step: %s, D Loss: %.3f, G Loss: %.3f, time: %.3f, total: %.3f'%(epoch, candi_epoch, step, D_loss.item(), G_loss.item(), time.time() - start,c_time))
                    
        #                print('x_outputs : %s, D_labels : %s'%(x_outputs.shape, D_labels.shape))
        #                print('z_outputs : %s, D_fakes : %s'%(z_outputs.shape, D_fakes.shape))
        
                    if step % 1000 == 0:
                        G.eval()
                        
                        img = get_sample_image(G, n_noise,DEVICE)
                        
                        sample_path=r'.\samples\0.8\{}_dataset\{}_class\{}_epoch'.format('MNIST',number,candi_epoch)
                        
                        if not os.path.exists(sample_path):
                            os.makedirs(sample_path)
                            
                            
                        imsave(sample_path+'\{}_step{}.jpg'.format(MODEL_NAME, str(step).zfill(3)), img, cmap='gray')
                        G.train()
                    step += 1
            
            model_path=r'.\models\0.8\%s_number\%s_epoch'%(number,candi_epoch)
            
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            torch.save(D.state_dict(), model_path+'\D.pkl')
            torch.save(G.state_dict(), model_path+'\G.pkl')
    
def get_sample_image(G, n_noise,DEVICE):
    """
        save sample 100 images
    """
    z = torch.randn(100, n_noise).to(DEVICE)
    y_hat = G(z).view(100, 28, 28) # (100, 28, 28)
    result = y_hat.cpu().data.numpy()
    img = np.zeros([280, 280])
    for j in range(10):
        img[j*28:(j+1)*28] = np.concatenate([x for x in result[j*10:(j+1)*10]], axis=-1)
    
    return img

    

#%%

if __name__ == '__main__':
    
    if not os.path.exists(r'.\samples'):
        os.makedirs('.\samples')
        
        
    if not os.path.exists(r'.\models'):
            os.makedirs('.\models')
            
    MODEL_NAME = 'VanillaGAN'
    #train_path=r'./data/untar/train/train/'
    n_cpu= multiprocessing.cpu_count()
    n_noise = 100
    
    batch_size=16

    main()


    