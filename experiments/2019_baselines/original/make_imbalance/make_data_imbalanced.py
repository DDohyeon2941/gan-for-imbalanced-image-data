# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 01:48:08 2019

@author: seonghee
"""

from IPython import display

# kill error when executing argparse in IPython console
import sys; sys.argv=['']; del sys

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from sampler import ImbalancedDatasetSampler

import copy
import torch
import torchvision.datasets
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch.nn.functional as F
import multiprocessing

def imbalanced_data(dataset,data_loader, cls_num,ratio,n_cpu,batch_size):
    
    
    idx_candi=np.where(data_loader.dataset.targets.numpy()==cls_num)[0]
    print(type(idx_candi))
    
    num_of_del=int(len(idx_candi)*ratio)
    print('num_of_del :', num_of_del)
    
    idx_=np.random.choice(idx_candi,num_of_del,replace=False)
    new_dataset=copy.deepcopy(dataset)

    new_dataset.targets = torch.from_numpy(np.delete(data_loader.dataset.targets.numpy(), idx_, axis=0))
    new_dataset.targets = torch.from_numpy(data_loader.dataset.targets.numpy()[idx_])
    
    new_dataset.data = torch.from_numpy(np.delete(data_loader.dataset.data.numpy(), idx_, axis=0))
    new_dataset.data = torch.from_numpy(data_loader.dataset.data.numpy()[idx_])
    
    new_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu,drop_last=True)
    
    
    num_list=[len(np.where(new_data_loader.dataset.targets.numpy()==x)[0]) for x in range(10) ]
    
    print('num_of_sample_class :', num_list)
    print('ratio_of_sample_class :', np.around(np.array(num_list)/60000,2))

    
    return new_dataset,new_data_loader

##
import matplotlib.pyplot as plt

#!pip install seaborn
import seaborn as sns

def show_mnist(arr, nrow=5, ncol=10, figsize=None):

    if figsize is None:
        figsize = (ncol, nrow)

    f, a = plt.subplots(nrow, ncol, figsize=figsize)

    def _do_show(the_figure, the_array):
        the_figure.imshow(the_array)
        the_figure.axis('off')

    for i in range(nrow):
        for j in range(ncol):
            _do_show(a[i][j], np.reshape(arr[i * ncol + j], (28, 28)))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.draw()

def vis(test_accs, confusion_mtxes, labels, figsize=(20, 8)):

    cm = confusion_mtxes[np.argmax(test_accs)]
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(test_accs, 'g')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.show()



#%%
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_cpu = multiprocessing.cpu_count()
    
    batch_size=64
    test_batch_size=64
    
    #load data
    test_seq = [
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]
    test_transform = torchvision.transforms.Compose(test_seq)
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(
            10, shear=10
        )] + test_seq)
    
    train_dataset = torchvision.datasets.MNIST('./download_data', train=True, download=True, transform=train_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu,drop_last=True)
    
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./download_data', train=False, transform=test_transform),
        batch_size=test_batch_size, shuffle=True, num_workers=n_cpu)

    imbalanced_dataset,imbalanced_data_loader=imbalanced_data(train_dataset, train_loader, 1, 0.3, 4, 64)



#%%
'''
    print('Original dataset: %d training samples & %d testing samples\n' %(len(imbalanced_data_loader.dataset), len(test_loader.dataset)))

    print('Distribution of classes in original dataset:')
    fig, ax = plt.subplots()
    _, counts = np.unique(imbalanced_data_loader.dataset.targets, return_counts=True)
    ax.bar(classe_labels, counts)
    ax.set_xticks(classe_labels)
    plt.show()
    
    for data, _ in imbalanced_data_loader:
        show_mnist(data)
        break
'''