import dataset_test
import torch
from model_test import LeNet5, CustomMLP
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torch.optim import SGD
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
import multiprocessing

# import some packages you need here


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
    model.train()
        
    running_loss = 0.0
    running_acc = 0.0
    for i,(inputs, labels) in enumerate(trn_loader):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.int64)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        predict=torch.max(outputs.data,1)[1]            
        accuracy=int(torch.sum(labels==predict)) / labels.size()[0]
        running_loss += loss.item()
        running_acc += accuracy

        # write your codes here
    trn_loss=running_loss/(i+1)
    acc=running_acc/(i+1)
    return trn_loss ,acc

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
    model.eval()
    # write your codes here
    running_loss = 0.0
    running_acc = 0.0
    for i, (inputs, labels) in enumerate(tst_loader):
        # get the inputs; data is a list of [inputs, labels]        
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.int64)

        # forward + backward + optimize
        outputs = model(inputs)        
        loss = criterion(outputs, labels)        
        predict=torch.max(outputs.data,1)[1]            
        accuracy=int(torch.sum(labels==predict)) / labels.size()[0]

        running_loss += loss.item()
        running_acc += accuracy

        
    tst_loss=running_loss/(i+1)
    acc=running_acc/(i+1)

    return tst_loss, acc

def main(epoch_num,save):
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    train_path1 = r'./data/untar/train/train/'
    train_path2=r'./oversampled_data/'
    
    test_path = r'./data/untar/test/test/'
    n_cpu= multiprocessing.cpu_count()

    train_batch=250
    Train_dataset_A = dataset_test.MNIST_(data_dir=train_path1)
    Train_dataset_B = dataset_test.MNIST_(data_dir=train_path2)
    concat_dataset = ConcatDataset((Train_dataset_A, Train_dataset_B))

    Train_loader= DataLoader(concat_dataset, batch_size=train_batch, shuffle=True,  num_workers=n_cpu)

    test_batch=125
    Test_dataset = dataset_test.MNIST_(data_dir=test_path)
    Test_loader= DataLoader(Test_dataset, batch_size=test_batch, shuffle=False,  num_workers=n_cpu)
    

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
    Lesnet=LeNet5().to(device)
    Custom=CustomMLP().to(device)
        
    cost_function = nn.CrossEntropyLoss()
    

    models=[Custom,Lesnet]

    bin_list=[]
    for model in models:
        
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

        train_loss=[]
        train_acc=[]
        vali_loss=[]
        vali_acc=[]
        
        for epoch in range(epoch_num):
            start = time.time()  # 시작 시간 저장
            trn_loss,trn_acc=train(model, Train_loader, device, cost_function, optimizer)
            tst_loss,tst_acc=test(model, Test_loader, device, cost_function)
    
            print('model : %s ,  %s epoch : , trn_loss : %.3f, trn_acc : %.3f, tst_loss : %.3f, tst_acc : %.3f, time: %.3f' %(str(model.__class__.__name__),epoch+1, trn_loss, trn_acc, tst_loss, tst_acc, (time.time() - start)))
            
            train_loss.append(trn_loss)
            train_acc.append(trn_acc)
            vali_loss.append(tst_loss)
            vali_acc.append(tst_acc)
            
        
        if save==True:
            if model.__class__.__name__ == 'LeNet5':
                df=pd.DataFrame({'LeNet_train_loss' : np.around(train_loss,3), 'LeNet_train_acc' : np.around(train_acc,3), 'LeNet_vali_loss' : np.around(vali_loss,3), 'LeNet_vali_acc' : np.around(vali_acc,3)})
            elif model.__class__.__name__ == 'CustomMLP':
                df=pd.DataFrame({'CustomMLP_train_loss' : np.around(train_loss,3), 'CustomMLP_train_acc' : np.around(train_acc,3), 'CustomMLP_vali_loss' : np.around(vali_loss,3), 'CustomMLP_vali_acc' : np.around(vali_acc,3)})
                
            bin_list.append(df)
            final_df = pd.concat(bin_list, axis=1)
            
            try:
                os.makedirs(r'./result/')
            except FileExistsError:
                pass
    
            final_df.to_csv(r'./result/result.csv', index=False)
    return final_df
#%%
if __name__ == '__main__':
    
    main(10,False)
