# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:07:17 2019

@author: User
"""

import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    
    """ LeNet-5 (LeCun et al., 1998)
        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """
    
    #Number of parameter: 61706
    
    def __init__(self):

        super(LeNet5, self).__init__()
        
        self.conv1= nn.Conv2d(1, 6, (5, 5), padding= 2)
        self.conv2= nn.Conv2d(6, 16, (5, 5))
        self.fc1= nn.Linear(16* 5* 5, 120)
        self.fc2= nn.Linear(120, 84)
        self.fc3= nn.Linear(84, 10)

        
    def forward(self, img):

        x= F.max_pool2d(F.relu(self.conv1(img)), (2, 2))
        x= F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x= x.view(-1, 16* 5* 5)
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        output= self.fc3(x)
        
        return output


class CustomMLP(nn.Module):
    
    """ Your custom MLP model
        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    #Number of parameter: 61710

    def __init__(self):

        super(CustomMLP, self).__init__()
        
        self.fc1= nn.Linear(28* 28, 64)
        self.fc2= nn.Linear(64, 64)
        self.fc3= nn.Linear(64, 64)
        self.fc4= nn.Linear(64, 42)
        self.fc5= nn.Linear(42, 10, bias= False)
        

    def forward(self, img):
        
        x= img.view(-1, 28* 28)
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= F.relu(self.fc3(x))
        x= F.relu(self.fc4(x))
        output= self.fc5(x)

        return output