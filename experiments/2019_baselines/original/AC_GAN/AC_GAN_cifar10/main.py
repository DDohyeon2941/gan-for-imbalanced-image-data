"""""""""
Pytorch implementation of Conditional Image Synthesis with Auxiliary Classifier GANs (https://arxiv.org/pdf/1610.09585.pdf).
This code is based on Deep Convolutional Generative Adversarial Networks in Pytorch examples : https://github.com/pytorch/examples/tree/master/dcgan
"""""""""
from __future__ import print_function
import argparse
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import model
import copy
import matplotlib.pyplot as plt



# cifar10
def imbalanced_data(dataset,data_loader, cls_num,ratio,n_cpu,batch_size):
    
    cls_num=cls_num.split('_')
    idx_candi=np.array([]).astype(int)

    for num_candi in cls_num:
        
        candi=np.where(torch.as_tensor(data_loader.dataset.targets).numpy() == int(num_candi))[0]
        print(candi)
        
        idx_candi=np.append(idx_candi,candi)
        
    print(idx_candi)
    
    num_of_del=int(len(idx_candi)*ratio)
    
    print('num_of_del :', num_of_del)
    
    idx_=np.random.choice(idx_candi,num_of_del,replace=False)
    
    new_dataset=copy.deepcopy(dataset)
    
#    new_dataset.targets = torch.from_numpy(np.delete(data_loader.dataset.targets.numpy(), idx_, axis=0))
    new_dataset.targets = torch.as_tensor(data_loader.dataset.targets).numpy()[idx_]
    
#    new_dataset.data = torch.from_numpy(np.delete(data_loader.dataset.data.numpy(), idx_, axis=0))
    new_dataset.data = torch.as_tensor(data_loader.dataset.data).numpy()[idx_]
    
    new_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu,drop_last=True)
    
    num_list=[len(np.where(torch.as_tensor(new_data_loader.dataset.targets).numpy()==x)[0]) for x in range(10) ]
    
    print('num_of_sample_class :', num_list)
    print('ratio_of_sample_class :', np.around(np.array(num_list)/len(dataset),2))

    return new_dataset,new_data_loader

#imbalanced_dataset,imbalanced_data_loader=imbalanced_data(train_dataset, dataloader, '1_2', 0.8, 0, 64)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='cifar10 | mnist',default='cifar10')
    parser.add_argument('--dataroot',  help='path to dataset', default='./datasets')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--niter', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--cls_num', help='list_of_numbers', default='3')
    #parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    opt = parser.parse_args()
    print(opt)
    print('-'*50)
    print('start'*30)
    print('-'*50)
    
    n_cpu= 0#multiprocessing.cpu_count()
    DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(opt.cls_num)
    print(type(opt.cls_num))
  
#    DEVICE='cuda:0'
    
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    
#    if opt.manualSeed is None:
#        pass
#        opt.manualSeed = random.randint(1, 10000)
#    print("Random Seed: ", opt.manualSeed)
    
#    random.seed(opt.manualSeed)
    
#    torch.manual_seed(opt.manualSeed)
 
    
#    if opt.cuda:
#        torch.cuda.manual_seed_all(opt.manualSeed)
    
    cudnn.benchmark = True
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    if opt.dataset == 'cifar10':
        train_dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    
    elif opt.dataset == 'MNIST':
        train_dataset = dset.MNIST(root=opt.dataroot, download=True, train=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])]))
    
    assert train_dataset
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=n_cpu,drop_last=True)
    
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    
    if opt.dataset == 'MNIST':
        nc = 1
        nb_label = 10
        
    else:
        nc = 3
        nb_label = 10
    
    netG = model.netG(nz, ngf, nc)
    netD = model.netD(ndf, nc, nb_label)
    
    
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    #print(netG)
    
    netD = model.netD(ndf, nc, nb_label)
    
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    #print(netD)
    
    s_criterion = nn.BCELoss()
    c_criterion = nn.NLLLoss()
    
    if opt.dataset =='MNIST':
        input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
    else:
        input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        
    real_label = 1
    fake_label = 0
    
    netD.to(DEVICE)
    netG.to(DEVICE)
    s_criterion.to(DEVICE)
    c_criterion.to(DEVICE)
    
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
    #test 함수가 있는 자리
    def test(predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)
    
    G_losses = []
    D_losses = []
    
    time_start=0
    #print(len(train_dataset), n_cpu, DEVICE)
    imbalanced_dataset,imbalanced_data_loader=imbalanced_data(train_dataset, dataloader, opt.cls_num, 0.8, n_cpu, opt.batchSize)
    
    for epoch in range(opt.niter):
        for i, data in enumerate(imbalanced_data_loader, 0):
            start=time.time()
            ##########################
            
            noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
            #print('noise shape : ' , noise.shape)
            fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
            #print('fixed noise shape :', fixed_noise.shape)
            s_label = torch.FloatTensor(opt.batchSize)
            c_label = torch.LongTensor(opt.batchSize)
            
            
            input, s_label = input.to(DEVICE), s_label.to(DEVICE)
            c_label = c_label.to(DEVICE)
            noise, fixed_noise = noise.to(DEVICE), fixed_noise.to(DEVICE)
            
            input = Variable(input)
            s_label = Variable(s_label)
            c_label = Variable(c_label)
            noise = Variable(noise)
            fixed_noise = Variable(fixed_noise)


            fixed_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
            random_label = np.random.randint(0, nb_label, opt.batchSize)
            #print('fixed label:{}'.format(random_label))
            

            random_onehot = np.zeros((opt.batchSize, nb_label))
            random_onehot[np.arange(opt.batchSize), random_label] = 1
            fixed_noise_[np.arange(opt.batchSize), :nb_label] = random_onehot[np.arange(opt.batchSize)]
    

            fixed_noise_ = torch.from_numpy(fixed_noise_)
            fixed_noise_ = fixed_noise_.resize_(opt.batchSize, nz, 1, 1)
            fixed_noise.data.copy_(fixed_noise_)
            
            ###########################
            # (1) Update D network
            ###########################
            # train with real
            netD.zero_grad()
            img, label = data
            
            if i==0:
                print(label)
            
            batch_size = img.size(0)
            input.data.resize_(img.size()).copy_(img).to(DEVICE)
            s_label.data.resize_(batch_size).fill_(real_label).to(DEVICE)
            c_label.data.resize_(batch_size).copy_(label).to(DEVICE)
            
            s_output, c_output = netD(input)
            s_errD_real = s_criterion(s_output, s_label).to(DEVICE)
            c_errD_real = c_criterion(c_output, c_label).to(DEVICE)
            errD_real = s_errD_real + c_errD_real
            errD_real.backward()
            D_x = s_output.data.mean()
            
            correct, length = test(c_output, c_label)
            # train with fake
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
    
            label = np.random.randint(0, nb_label, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            label_onehot = np.zeros((batch_size, nb_label))
            label_onehot[np.arange(batch_size), label] = 1
            noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
            
            noise_ = (torch.from_numpy(noise_)).to(DEVICE)
            noise_ = noise_.resize_(batch_size, nz, 1, 1).to(DEVICE)
            noise.data.copy_(noise_).to(DEVICE)
    
            c_label.data.resize_(batch_size).copy_(torch.from_numpy(label)).to(DEVICE)
    
            fake = netG(noise).to(DEVICE)
            #print('fake_shape:', np.arary(fake).shape)
            #print('fake_min:', np.min(np.arary(fake).shape))
            s_label.data.fill_(fake_label)
            s_output,c_output = netD(fake.detach())
            s_errD_fake = s_criterion(s_output, s_label)
            c_errD_fake = c_criterion(c_output, c_label)
            errD_fake = s_errD_fake + c_errD_fake
    
            errD_fake.backward()
            D_G_z1 = s_output.data.mean()
            errD = s_errD_real + s_errD_fake
            optimizerD.step()
    
            ###########################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            s_label.data.fill_(real_label)  # fake labels are real for generator cost
            s_output,c_output = netD(fake)
            s_errG = s_criterion(s_output, s_label)
            c_errG = c_criterion(c_output, c_label)
            
            errG = s_errG + c_errG
            errG.backward()
            D_G_z2 = s_output.data.mean()
            optimizerG.step()
            
            time_start += time.time() - start

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, Accuracy: %.4f / %.4f = %.4f, time : %.4f, c_time=%.4f'
                  
                  # 변경해야함
                  % (epoch, opt.niter, i, len(imbalanced_data_loader),
                     errD.data, errG.data, D_x, D_G_z1, D_G_z2,
                     correct, length, 100.* correct / length, time.time() - start, time_start))
            
            # MNIST
            #file_path=r'./samples/M_changed_seed_50'
            
            #cifar10
            file_path=r'./samples/cifar10-test_image'
            
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            
            if i % 100 == 0:
                #fake = netG(fixed_cat)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data,
                        file_path + '/%s_epoch %s_batch_fake_samples.png' % (epoch,i))
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())
    
    model_path=r'./models'
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    torch.save(netD.state_dict(), model_path+'/D2.pkl')
    torch.save(netG.state_dict(), model_path+'/G2.pkl')
    
## 그래프용 ##
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
            

    
    
    