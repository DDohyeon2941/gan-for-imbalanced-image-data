# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:07:56 2019

@author: User
"""

import dataset
from model import LeNet5, CustomMLP
import tarfile
import os
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import multiprocessing
import pandas as pd


def train(model, trn_loader, device, criterion, optimizer):
    
    """ Train function
    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim
    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    running_loss, correct, total= 0, 0, 0
        
    for num_batch_iter, (x, target) in enumerate(trn_loader):
        
        x, target= x.to(device), target.to(device)
        
        optimizer.zero_grad()
        model_output= model(x)
        
        loss= criterion(model_output, target)
        loss.backward()
        optimizer.step()
        
        model_output= model(x)
        _, pred= torch.max(model_output.data, 1)
        
        correct+= (pred== target).sum().item()
        running_loss+= loss.item()
        total+= target.size(0)
        
    trn_loss= running_loss/ (num_batch_iter+ 1)
    acc= correct/ total
    
    return trn_loss, acc


def test(model, tst_loader, device, criterion):
    
    """ Test function
    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function
    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    loss, correct, total= 0, 0, 0
    
    with torch.no_grad():
        
        for num_batch_iter, (x, target) in enumerate(tst_loader):
            
            x, target= x.to(device), target.to(device)
            
            model_output= model(x)
            
            loss+= criterion(model_output, target).item()
            _, pred= torch.max(model_output.data, 1)
            total+= target.size(0)
            correct+= (pred== target).sum().item()
            
    tst_loss= loss/ (num_batch_iter+ 1)
    acc= correct/ total

    return tst_loss, acc


def main():
    
    """ Main function
        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss
    """
    
    n_cpu= multiprocessing.cpu_count()
    
    data_dir_dict= {'train': '../data/train', 'test': '../data/test'}
    
    for _, data_dir in data_dir_dict.items():
        
        if not os.path.isdir(data_dir):
            
            tar= tarfile.open('%s.tar'%data_dir, mode= 'r')
            tar.extractall(path= './data/')


    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs= 50
    batch_size = 512

    criterion= nn.CrossEntropyLoss()
    
    train_dataset= dataset.MNIST(data_dir_dict['train'])
    train_loader= DataLoader(dataset= train_dataset, 
                             batch_size= batch_size,
                             shuffle= True,
                             num_workers= n_cpu)

    test_dataset= dataset.MNIST(data_dir_dict['test'])
    test_loader= DataLoader(dataset= test_dataset,
                            batch_size= batch_size,
                            shuffle= False, 
                            num_workers= n_cpu)
    
    
    net= LeNet5().to(device)
    net_optimizer= SGD(net.parameters(), lr= .01, momentum= .9)
    
    net_trn_loss_list, net_trn_acc_list, net_tst_loss_list, net_tst_acc_list= [], [], [], []
    
    print('--------------LeNet5--------------')
    for epoch in range(num_epochs):
        
        net_time= time.time()
        
        trn_loss, trn_acc= train(net, train_loader, device, criterion, net_optimizer)
        tst_loss, tst_acc= test(net, test_loader, device, criterion)
        
        print('%s epoch || %.5f time'%((epoch+ 1), time.time()- net_time))
        print('avg train loss: %.5f | avg train acc: %.5f ||| avg test loss: %.5f | avg test acc: %.5f'\
              %(trn_loss, trn_acc, tst_loss, tst_acc))
        print('')
        
        net_trn_loss_list.append(trn_loss)
        net_trn_acc_list.append(trn_acc)
        net_tst_loss_list.append(tst_loss)
        net_tst_acc_list.append(tst_acc)
        
    
    mlp= CustomMLP().to(device)
    mlp_optimizer= SGD(mlp.parameters(), lr= .01, momentum= .9)
    
    mlp_trn_loss_list, mlp_trn_acc_list, mlp_tst_loss_list, mlp_tst_acc_list= [], [], [], []
    
    print('--------------CustomMLP--------------')        
    for epoch in range(num_epochs):
        
        mlp_time= time.time()
        
        trn_loss, trn_acc= train(mlp, train_loader, device, criterion, mlp_optimizer)
        tst_loss, tst_acc= test(mlp, test_loader, device, criterion)

        print('%s epoch || %.5f time'%((epoch+ 1), time.time()- mlp_time))
        print('avg train loss: %.5f | avg train acc: %.5f ||| avg test loss: %.5f | avg test acc: %.5f'\
              %(trn_loss, trn_acc, tst_loss, tst_acc))
        print('')
        
        mlp_trn_loss_list.append(trn_loss)
        mlp_trn_acc_list.append(trn_acc)
        mlp_tst_loss_list.append(tst_loss)
        mlp_tst_acc_list.append(tst_acc)

    
    del train_dataset
    del train_loader
    
    train_dataset_aug= dataset.MNIST(data_dir_dict['train'], aug_option= True)
    train_loader_aug= DataLoader(dataset= train_dataset_aug, 
                                 batch_size= batch_size,
                                 shuffle= True,
                                 num_workers= n_cpu)
    
    net_aug= LeNet5().to(device)
    net_optimizer_aug= SGD(net_aug.parameters(), lr= .01, momentum= .9)

    net_aug_trn_loss_list, net_aug_trn_acc_list, net_aug_tst_loss_list, net_aug_tst_acc_list= [], [], [], []
    
    print('--------------LeNet5 with augmentation--------------')
    for epoch in range(num_epochs):
        
        net_time= time.time()
        
        trn_loss, trn_acc= train(net_aug, train_loader_aug, device, criterion, net_optimizer_aug)
        tst_loss, tst_acc= test(net_aug, test_loader, device, criterion)
        
        print('%s epoch || %.5f time'%((epoch+ 1), time.time()- net_time))
        print('avg train loss: %.5f | avg train acc: %.5f ||| avg test loss: %.5f | avg test acc: %.5f'\
              %(trn_loss, trn_acc, tst_loss, tst_acc))
        print('')
        
        net_aug_trn_loss_list.append(trn_loss)
        net_aug_trn_acc_list.append(trn_acc)
        net_aug_tst_loss_list.append(tst_loss)
        net_aug_tst_acc_list.append(tst_acc)
            
        
    mlp_aug= CustomMLP().to(device)
    mlp_optimizer_aug= SGD(mlp_aug.parameters(), lr= .01, momentum= .9)

    mlp_aug_trn_loss_list, mlp_aug_trn_acc_list, mlp_aug_tst_loss_list, mlp_aug_tst_acc_list= [], [], [], []
    
    print('--------------CustomMLP with augmentation--------------')        
    for epoch in range(num_epochs):
        
        mlp_time= time.time()
        
        trn_loss, trn_acc= train(mlp_aug, train_loader_aug, device, criterion, mlp_optimizer_aug)
        tst_loss, tst_acc= test(mlp_aug, test_loader, device, criterion)

        print('%s epoch || %.5f time'%((epoch+ 1), time.time()- mlp_time))
        print('avg train loss: %.5f | avg train acc: %.5f ||| avg test loss: %.5f | avg test acc: %.5f'\
              %(trn_loss, trn_acc, tst_loss, tst_acc))
        print('')
        
        mlp_aug_trn_loss_list.append(trn_loss)
        mlp_aug_trn_acc_list.append(trn_acc)
        mlp_aug_tst_loss_list.append(tst_loss)
        mlp_aug_tst_acc_list.append(tst_acc)

        
    net_result, net_aug_result, mlp_result, mlp_aug_result= {}, {}, {}, {}
    
    net_result['train_loss']= net_trn_loss_list
    net_result['train_acc']= net_trn_acc_list
    net_result['test_loss']= net_tst_loss_list
    net_result['test_acc']= net_tst_acc_list
    
    net_aug_result['train_loss']= net_aug_trn_loss_list
    net_aug_result['train_acc']= net_aug_trn_acc_list
    net_aug_result['test_loss']= net_aug_tst_loss_list
    net_aug_result['test_acc']= net_aug_tst_acc_list
    
    mlp_result['train_loss']= mlp_trn_loss_list
    mlp_result['train_acc']= mlp_trn_acc_list
    mlp_result['test_loss']= mlp_tst_loss_list
    mlp_result['test_acc']= mlp_tst_acc_list
    
    mlp_aug_result['train_loss']= mlp_aug_trn_loss_list
    mlp_aug_result['train_acc']= mlp_aug_trn_acc_list
    mlp_aug_result['test_loss']= mlp_aug_tst_loss_list
    mlp_aug_result['test_acc']= mlp_aug_tst_acc_list

    
    net_result= pd.DataFrame(net_result, columns= ['train_loss', 'train_acc',
                                                   'test_loss', 'test_acc'])
    
    net_aug_result= pd.DataFrame(net_aug_result, columns= ['train_loss', 'train_acc',
                                                           'test_loss', 'test_acc'])
    
    mlp_result= pd.DataFrame(mlp_result, columns= ['train_loss', 'train_acc',
                                                   'test_loss', 'test_acc'])
    
    mlp_aug_result= pd.DataFrame(mlp_aug_result, columns= ['train_loss', 'train_acc',
                                                           'test_loss', 'test_acc'])
    
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    
    net_result.to_csv('../results/LeNet5.csv', index= False)
    net_aug_result.to_csv('../results/LeNet5_aug.csv', index= False)
    mlp_result.to_csv('../results/MLP.csv', index= False)
    mlp_aug_result.to_csv('../results/MLP_aug.csv', index= False)
    
    ######LeNet known accuracy: 99.21%
    
    #LeNet train accuracy: 99.98%
    #LeNet test accuracy: 99.05%
    #LeNet with augmentation train accuracy: 100%
    #LeNet with augmentation train accuracy: 99.19%
    #MLP train accuracy: 99.99%
    #MLP test accuracy:  97.34%
    #MLP with augmentation train accuracy: 100%
    #MLP with augmentation train accuracy: 97.33%

if __name__ == '__main__':
    
    main()