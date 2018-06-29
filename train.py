#coding=utf-8
from __future__ import print_function
import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from layer import discriminator, CompletionNet
import torchvision as tv
from PIL import Image
import numpy as np

def data_load(image_path, batch_size, num_workers):
    transforms = tv.transforms.Compose([
                    tv.transforms.Resize(256),
                    tv.transforms.CenterCrop(256),
                    tv.transforms.ToTensor(),
                    #tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    dataset= tv.datasets.ImageFolder(image_path, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            drop_last=True)
    return dataloader



def train(opt):
    # define the net
    net_g = CompletionNet()
    net_d = discriminator()
    #net_d_crop = Discriminator1()
    loss_net = []
    loss_epoch = 0.0
    i_n = 0

    # define optimizer
    optimizer_g = torch.optim.Adadelta(net_g.parameters(), lr=0.01, rho=0.95)
    optimizer_d = torch.optim.Adadelta(net_d.parameters(), lr=0.01, rho=0.95)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, 50, gamma=0.1, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, 50, gamma=0.1, last_epoch=-1)
    # define loss function
    loss_g = nn.MSELoss()
    #loss_d = nn.CrossEntropyLoss()
    #loss_d = nn.BCELoss()
    loss_d = nn.MSELoss()
    
    # load the data, and reshape to 256*256
    data_in = data_load(opt.data_path, opt.batch_size, opt.num_workers)
    if opt.use_gpu:
        net_g = net_g.cuda()
        net_d = net_d.cuda()
        loss_g = loss_g.cuda()
        loss_d = loss_d.cuda()

    
    print('begin trainning')
    for i in range(opt.max_epoch):
        print('epoch:' + str(i))
        torch.backends.cudnn.benchmark=True
        for ii, (img, _) in enumerate(data_in):
            
            #convert to PILImages and crop and then convert to tensor
#             transforms1 = tv.transforms.Compose([
#                 tv.transforms.ToTensor(),
#                 ]
#             )
            
            img_raw = Variable(img, requires_grad=False)
            img_raw = img_raw.cuda()
            
            temp_pic = img.clone()
            #temp_pic = tv.transforms.ToPILImage()(temp_pic)
                
            #region = (128, 128, 384, 384)
            #print(type(temp_pic))
            #crop_pic = temp_pic.crop(region)
            temp_pic = temp_pic.data.cpu().numpy()
            #print(temp_pic.shape)
            crop_pic = temp_pic[:,:,64:192, 64:192]
            
            #print(crop_pic.shape)
            crop_pic = torch.from_numpy(crop_pic)
            
            #crop_pic = transforms1(crop_pic)
            crop_pic = Variable(crop_pic, requires_grad=False)
            crop_pic = crop_pic.cuda()
            # raw data input
          
            
#             if opt.use_gpu:
#                 img_raw = img_raw.cuda()


            img_in = crop_pic
            img_in = img_in.cuda()
            #print(img_in.size())
            # generate a center area at the first epoch
          




               
                
            if i%4<opt.c_epoch:
                optimizer_g.zero_grad()
                img_g_out_raw = net_g(img_in)
                img_g_out_np = img_g_out_raw.data.cpu().numpy()
                
                
                #print(img_g_out_np.shape)
                
                img_g_out_crop = img_g_out_np[:,:,64:192, 64:192]
                img_g_out_crop = torch.from_numpy(img_g_out_crop)
                img_g_out_crop = Variable(img_g_out_crop, requires_grad=False)
                img_g_out_crop = img_g_out_crop.cuda()
                

                error_c = loss_g(img_g_out_crop, img_in)
                error_g = loss_g(img_g_out_raw, img_raw)
                error_G = error_c + error_g
                error_G.backward()
                optimizer_g.step()
                scheduler_g.step()

            # if opt.c_epoch<=i%5 and i%5<(opt.c_epoch + opt.d_epoch):
            else:
                optimizer_d.zero_grad()
                img_g_out_raw = net_g(img_in)
                
                
                img_g_out_np = img_g_out_raw.data.cpu().numpy()*255
                
                img_g_out_crop = img_g_out_np[:,:,64:192, 64:192]
                img_g_out_crop = torch.from_numpy(img_g_out_crop)
                img_g_out_crop = Variable(img_g_out_crop, requires_grad=False)
                img_g_out_crop = img_g_out_crop.cuda()

                real_score1 = net_d(img_raw)
                real_score1 = torch.squeeze(real_score1)
                real_score1 = Variable(real_score1, requires_grad=True)
                fake_score1 = net_d(img_g_out_raw)
                fake_score1 = torch.squeeze(fake_score1)
                fake_score1 = Variable(fake_score1, requires_grad=True)
                #error_d1 = loss_d(real_score1, fake_score1)
                a = Variable(torch.ones(64), requires_grad=False)
                a = a.cuda()
                b = Variable(torch.zeros(64), requires_grad=False)
                b = b.cuda()
                #print(real_score1.shape)
                #print(a.shape)
                error_1a = loss_d(real_score1, a)
                error_2a = loss_d(fake_score1, b)

                real_score2 = net_d(img_in)
                #print(real_score2.shape)
                real_score2 = torch.squeeze(real_score2)
                real_score2 = Variable(real_score2, requires_grad=True)
                fake_score2 = net_d(img_g_out_crop)
                fake_score2 = torch.squeeze(fake_score2)
                fake_score2 = Variable(fake_score2, requires_grad=True)
                
                c = Variable(torch.ones(16), requires_grad=False)
                c = c.cuda()
                d = Variable(torch.zeros(16), requires_grad=False)
                d = d.cuda()
                error_1b = loss_d(real_score2, c)
                error_2b = loss_d(fake_score2, d)

                error_1 = error_1a + error_1b
                error_2 = error_2a + error_2b
                
                error_1.backward()
                error_2.backward()
                optimizer_d.step()
                scheduler_d.step()

                if i%4 >= (opt.c_epoch + opt.d_epoch):
                    # optimizer_c = torch.optim.Adadelta(net_c.parameters(), rho=0.95)
                    optimizer_g.zero_grad()
                    img_g_out_raw = net_g(img_in)
                    
                    img_g_out_np = img_g_out_raw.data.cpu().numpy()
                    
                    img_g_out_crop = img_g_out_np[:,:,64:192, 64:192]
                    img_g_out_crop = torch.from_numpy(img_g_out_crop)
                    img_g_out_crop = Variable(img_g_out_crop, requires_grad=False)
                    img_g_out_crop = img_g_out_crop.cuda()

                    real_score1 = net_d(img_raw)
                    real_score1 = torch.squeeze(real_score1)
                    real_score1 = Variable(real_score1, requires_grad=True)
                    fake_score1 = net_d(img_g_out_raw)
                    fake_score1 = torch.squeeze(fake_score1)
                    fake_score1 = Variable(fake_score1, requires_grad=True)
                    
                    a = Variable(torch.ones(64), requires_grad=False)
                    a = a.cuda()
                    b = Variable(torch.zeros(64), requires_grad=False)
                    b = b.cuda()
                    error_1a = loss_d(real_score1, a)
                    error_2a = loss_d(fake_score1, b)
  
                    real_score2 = net_d(img_in)
                    real_score2 = torch.squeeze(real_score2)
                    real_score2 = Variable(real_score2, requires_grad=True)
                    fake_score2 = net_d(img_g_out_crop)
                    fake_score2 = torch.squeeze(fake_score2)
                    fake_score2 = Variable(fake_score2, requires_grad=True)
                    
                    c = Variable(torch.ones(16), requires_grad=False)
                    c = c.cuda()
                    d = Variable(torch.zeros(16), requires_grad=False)
                    d = d.cuda()
                    error_1b = loss_d(real_score2, c)
                    error_2b = loss_d(fake_score2, d)
 
                    error_1 = error_1a + error_1b
                    error_2 = error_2a + error_2b

                    error_c = loss_g(img_g_out_crop, img_in)
                    error_g = loss_g(img_g_out_raw, img_raw)
                    error_G = error_c + error_g


                    error_1.backward()
                    error_2.backward()
                    error_G.backward()
                    optimizer_g.step()
                    scheduler_g.step()


                    print('error_G:%f, error_c1:%f, error_c2:%f, ii:%d'  %(error_G, error_1, error_2, ii))

                    # if(i>i_n):
                    #     loss_net.append(loss_epoch)
                    #     i_n = i
                    #     loss_epoch = 0
            if (i+1)%opt.save_epoch==0:
                # print(img_c_out)
                tv.utils.save_image(img_g_out_raw.data, '%s/%s.png' %(opt.save_path, ii))
                tv.utils.save_image(img_in.data, '%s/%s_in.png' %(opt.save_path, ii))
                torch.save(net_g.state_dict(), './checkpoints/net_g_%s.pth' %i)
                torch.save(net_d.state_dict(), './checkpoints/net_d_%s.pth' %i)
                optimizer_g = torch.optim.Adadelta(net_g.parameters(), rho=0.95)
                optimizer_d = torch.optim.Adadelta(net_d.parameters(), rho=0.95)
