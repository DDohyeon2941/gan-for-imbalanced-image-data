import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,stride=1, bias=True)
        self.conv2=nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=True)
        self.fc1=nn.Linear(16*5*5, 120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

        # write your codes here

    def forward(self, img):

        # write your codes here
        out=F.max_pool2d(F.relu(self.conv1(img)), (2,2))
        out=F.max_pool2d(F.relu(self.conv2(out)), (2,2))
        out=out.view(-1, 16*5*5)
        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        output=self.fc3(out)
        
        return output

class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(32*32,58)
        self.fc2 = nn.Linear(58,33)
        self.fc3 = nn.Linear(33,10)
        # write your codes here

    def forward(self, img):

        img=img.view(-1,32*32)
        out=F.relu(self.fc1(img))
        out=F.relu(self.fc2(out))
        output=self.fc3(out)

        # write your codes here
        return output
    


