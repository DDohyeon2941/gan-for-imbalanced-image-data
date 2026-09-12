# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:08:38 2019

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt

def curve_plot_vs(LeNet5_dir, MLP_dir, save_option= True):
    
    '''
    Function for comparision between two models
    '''
    
    net= pd.read_csv(LeNet5_dir)
    mlp= pd.read_csv(MLP_dir)
    
    net_trn_loss, net_trn_acc= net['train_loss'], net['train_acc']
    net_tst_loss, net_tst_acc= net['test_loss'], net['test_acc']
    
    mlp_trn_loss, mlp_trn_acc= mlp['train_loss'], mlp['train_acc']
    mlp_tst_loss, mlp_tst_acc= mlp['test_loss'], mlp['test_acc']
    
    idx= net.index
    net_acc_argmax= net_tst_acc.values.argmax()
    mlp_acc_argmax= mlp_tst_acc.values.argmax()
    maxpoint= [(net_acc_argmax, net_tst_acc[net_acc_argmax]),
               (mlp_acc_argmax, mlp_tst_acc[mlp_acc_argmax])]
    
    fig= plt.figure(figsize= (9, 9))
    plt.suptitle('LeNet5 vs CustomMLP', fontsize= 16)
    plt.subplots_adjust(wspace= .3, hspace= .3)

    
    ax1= fig.add_subplot(2, 2, 1)
    ax1.grid(True)
    ax1.plot(idx, net_trn_loss, 'r', label= 'LeNet5 train loss')
    ax1.plot(idx, mlp_trn_loss, 'g', label= 'MLP train loss')
    ax1.legend(loc= 'upper right')
    ax1.set_ylim(top= .5)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('Train loss curve LeNet5 vs MLP', fontsize= 10)
    
    ax2= fig.add_subplot(2, 2, 2)
    ax2.grid(True)
    ax2.plot(idx, net_tst_loss, 'r', label= 'LeNet5 test loss')
    ax2.plot(idx, mlp_tst_loss, 'g', label= 'MLP test loss')
    ax2.legend(loc= 'upper right')
    ax2.set_ylim(top= .5)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('Test loss curve LeNet5 vs MLP', fontsize= 10)
    
    ax3= fig.add_subplot(2, 2, 3)
    ax3.grid(True)
    ax3.plot(idx, net_trn_acc, 'r', label= 'LeNet5 train accuracy')
    ax3.plot(idx, mlp_trn_acc, 'g', label= 'MLP train accuracy')
    ax3.legend(loc= 'lower right')
    ax3.set_ylim(bottom= .85)
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('acc')
    ax3.set_title('Train accuracy curve LeNet5 vs MLP', fontsize= 10)
    
    ax4= fig.add_subplot(2, 2, 4)
    ax4.grid(True)
    ax4.plot(idx, net_tst_acc, 'r', label= 'LeNet5 test accuracy')
    ax4.plot(idx, mlp_tst_acc, 'g', label= 'MLP test accuracy')
    ax4.legend(loc= 'lower right')
    ax4.set_ylim(bottom= .85)
    ax4.set_xlabel('epoch')
    ax4.set_ylabel('acc')
    ax4.set_title('Test accuracy curve LeNet5 vs MLP', fontsize= 10)
    ax4.plot(net_acc_argmax, net_tst_acc[net_acc_argmax], 'r', marker= 'o')
    ax4.plot(mlp_acc_argmax, mlp_tst_acc[mlp_acc_argmax], 'g', marker= 'o')
    for (idxs, value) in maxpoint:
        ax4.annotate('%.5f is maximum acc'%value, (idxs, value),
                     xytext= (-60, -20),
                     textcoords= 'offset points', 
                     arrowprops= {'arrowstyle': '->'})
    
    fig.savefig('../results/result_curve_vs.png', dpi= 300)
    
    
def curve_plot_tv(LeNet5_dir, MLP_dir, save_option= True):
    
    '''
    Function for comparision between train set and test set
    '''
    
    net= pd.read_csv(LeNet5_dir)
    mlp= pd.read_csv(MLP_dir)
    
    net_trn_loss, net_trn_acc= net['train_loss'], net['train_acc']
    net_tst_loss, net_tst_acc= net['test_loss'], net['test_acc']
    
    mlp_trn_loss, mlp_trn_acc= mlp['train_loss'], mlp['train_acc']
    mlp_tst_loss, mlp_tst_acc= mlp['test_loss'], mlp['test_acc']
    
    idx= net.index
    
    fig= plt.figure(figsize= (9, 9))
    plt.suptitle('Comparision between train set and test set in one model', fontsize= 16)
    plt.subplots_adjust(wspace= .3, hspace= .3)
    
    ax1= fig.add_subplot(2, 2, 1)
    ax1.grid(True)
    ax1.plot(idx, net_trn_loss, 'r', label= 'LeNet5 train loss')
    ax1.plot(idx, net_tst_loss, 'g', label= 'LeNet5 test loss')
    ax1.legend(loc= 'upper right')
    ax1.set_ylim(top= .5)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('LeNet5 Loss Comparision between Train and Test', fontsize= 10)
    
    ax2= fig.add_subplot(2, 2, 3)
    ax2.grid(True)
    ax2.plot(idx, mlp_trn_loss, 'r', label= 'MLP train loss')
    ax2.plot(idx, mlp_tst_loss, 'g', label= 'MLP test loss')
    ax2.legend(loc= 'upper right')
    ax2.set_ylim(top= .5)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('MLP loss comparision between train and test', fontsize= 10)
    
    ax3= fig.add_subplot(2, 2, 2)
    ax3.grid(True)
    ax3.plot(idx, net_trn_acc, 'r', label= 'LeNet5 train accuracy')
    ax3.plot(idx, net_tst_acc, 'g', label= 'LeNet5 test accuracy')
    ax3.legend(loc= 'lower right')
    ax3.set_ylim(bottom= .85)
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('acc')
    ax3.set_title('LeNet5 accuracy comparision between train and test', fontsize= 10)
    
    ax4= fig.add_subplot(2, 2, 4)
    ax4.grid(True)
    ax4.plot(idx, mlp_trn_acc, 'r', label= 'MLP train accuracy')
    ax4.plot(idx, mlp_tst_acc, 'g', label= 'MLP test accuracy')
    ax4.legend(loc= 'lower right')
    ax4.set_ylim(bottom= .85)
    ax4.set_xlabel('epoch')
    ax4.set_ylabel('acc')
    ax4.set_title('MLP accuracy comparision between train and test', fontsize= 10)
    
    fig.savefig('../results/result_curve_tv.png', dpi= 300)


def curve_plot_aug(model_dir, model_aug_dir, save_option= True):
    
    '''
    '''
    
    model= pd.read_csv(model_dir)
    model_name= model_dir.split('/')[-1][:-4]
    model_aug= pd.read_csv(model_aug_dir)
    
    model_trn_loss, model_trn_acc= model['train_loss'], model['train_acc']
    model_tst_loss, model_tst_acc= model['test_loss'], model['test_acc']
    
    model_aug_trn_loss, model_aug_trn_acc= model_aug['train_loss'], model_aug['train_acc']
    model_aug_tst_loss, model_aug_tst_acc= model_aug['test_loss'], model_aug['test_acc']
    
    idx= model.index
    model_acc_argmax= model_tst_acc.values.argmax()
    model_aug_acc_argmax= model_aug_tst_acc.values.argmax()
    maxpoint= [(model_aug_acc_argmax, model_aug_tst_acc[model_aug_acc_argmax]),
               (model_acc_argmax, model_tst_acc[model_acc_argmax])]

    
    fig= plt.figure(figsize= (9, 9))
    plt.suptitle('%s loss and accuracy comparision with augmentation'%model_name, fontsize= 16)
    plt.subplots_adjust(wspace= .3, hspace= .3)
    
    ax1= fig.add_subplot(2, 2, 1)
    ax1.grid(True)
    ax1.plot(idx, model_trn_loss, 'g', label= '%s'%model_name)
    ax1.plot(idx, model_aug_trn_loss, 'r', label= '%s with augmentation'%model_name)
    ax1.legend(loc= 'upper right')
    ax1.set_ylim(top= .5)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('%s Loss Comparision with augmentation'%model_name, fontsize= 10)
    
    ax2= fig.add_subplot(2, 2, 3)
    ax2.grid(True)
    ax2.plot(idx, model_tst_loss, 'g', label= '%s'%model_name)
    ax2.plot(idx, model_aug_tst_loss, 'r', label= '%s with augmentation'%model_name)
    ax2.legend(loc= 'upper right')
    ax2.set_ylim(top= .5)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('%s loss comparision with augmentation'%model_name, fontsize= 10)
    
    ax3= fig.add_subplot(2, 2, 2)
    ax3.grid(True)
    ax3.plot(idx, model_trn_acc, 'g', label= '%s'%model_name)
    ax3.plot(idx, model_aug_trn_acc, 'r', label= '%s with augmentation'%model_name)
    ax3.legend(loc= 'lower right')
    ax3.set_ylim(bottom= .85)
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('acc')
    ax3.set_title('%s accuracy comparision with augmentation'%model_name, fontsize= 10)
    
    ax4= fig.add_subplot(2, 2, 4)
    ax4.grid(True)
    ax4.plot(idx, model_tst_acc, 'g', label= '%s'%model_name)
    ax4.plot(idx, model_aug_tst_acc, 'r', label= '%s with augmentation'%model_name)
    ax4.legend(loc= 'lower right')
    ax4.set_ylim(bottom= .85)
    ax4.set_xlabel('epoch')
    ax4.set_ylabel('acc')
    ax4.set_title('%s accuracy comparision with augmentation'%model_name, fontsize= 10)
    ax4.plot(model_acc_argmax, model_tst_acc[model_acc_argmax], 'g', marker= 'o')
    ax4.plot(model_aug_acc_argmax, model_aug_tst_acc[model_aug_acc_argmax], 'r', marker= 'o')
    for i, (idxs, value) in enumerate(maxpoint):
        ax4.annotate('%.5f is maximum acc'%value, (idxs, value),
                     xytext= (-60, -20- (15* i)),
                     textcoords= 'offset points', 
                     arrowprops= {'arrowstyle': '->'})

    
    fig.savefig('../results/%s_result_curve_aug.png'%model_name, dpi= 300)
    

    
if __name__== '__main__':
    
    net_result= '../results/LeNet5.csv'
    mlp_result= '../results/MLP.csv'
    net_aug_result= '../results/LeNet5_aug.csv'
    mlp_aug_result= '../results/MLP_aug.csv'
    
    curve_plot_vs(net_result, mlp_result, save_option= True)
    curve_plot_tv(net_result, mlp_result, save_option= True)
    curve_plot_aug(net_result, net_aug_result, save_option= True)
    curve_plot_aug(mlp_result, mlp_aug_result, save_option= True)