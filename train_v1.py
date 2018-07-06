#coding=utf-8
from __future__ import print_function
import os
import sys


import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from layer_v1 import CompletionNet, discriminator
import torchvision as tv
from PIL import Image
import numpy as np

from load_data import DatasetFromFolder







# def data_load(image_path, batch_size, num_workers):
#     transforms = tv.transforms.Compose([
#                     #tv.transforms.Resize(256),
#                     #tv.transforms.CenterCrop(256),
#                     tv.transforms.ToTensor(),
#                     #tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                     ])
#     dataset= tv.datasets.ImageFolder(image_path, transform=transforms)
#     dataloader = torch.utils.data.DataLoader(dataset,
#                                             batch_size=batch_size,
#                                             shuffle=True,
#                                             num_workers=num_workers,
#                                             drop_last=True)
#     return dataloader



def train(opt):
    # define the net
    net_c = CompletionNet()
    net_d = discriminator()
    loss_net = []
    loss_epoch = 0.0
    i_n = 0

    # define optimizer
    optimizer_c = torch.optim.Adadelta(net_c.parameters(), rho=0.95)
    optimizer_d = torch.optim.Adadelta(net_d.parameters(), rho=0.95)
    scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, 50, gamma=0.1, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, 50, gamma=0.1, last_epoch=-1)
    # define loss function
    loss_c = nn.MSELoss()
    # loss_d = nn.CrossEntropyLoss()
    loss_d = nn.MSELoss()

    

    dataset = DatasetFromFolder(opt.data_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.num_workers,
                                             drop_last=True)





    
    if opt.use_gpu:
        net_c = torch.nn.DataParallel(net_c)
        net_c.cuda()
        net_d = torch.nn.DataParallel(net_d)
        net_d.cuda()
        loss_c.cuda()
        loss_d.cuda()

    image_lenght_list = []
    image_height_list = []
    len_begin_list = []
    hei_begin_list = []
    len_v_list = []
    hei_v_list = []
    len_m_list = []
    hei_m_list = []
    len_mask_m_list = []
    hei_mask_m_list = []
    print('begin trainning')
    for i in range(opt.max_epoch):
        print('epoch:' + str(i))
        for ii, (img, target) in enumerate(dataloader):
            # raw data input
            img_raw = Variable(img)
            target = Variable(target)
            if opt.use_gpu:
                img_raw = img_raw.cuda()
                target = target.cuda()



            # mask the photo input
            # img_in = img_raw.clone()

            # temp_pic = img_in.cpu().numpy()
            # #print(temp_pic.shape)
            # crop_pic = temp_pic[:,:,28:228, 28:228]

            # #print(crop_pic.shape)
            # crop_pic = torch.from_numpy(crop_pic)

            # #crop_pic = transforms1(crop_pic)
            # img_in = Variable(crop_pic, requires_grad=False)
            # img_in = img_in.cuda()







            # if opt.use_gpu:
            #     mask_c = torch.cuda.FloatTensor(opt.batch_size, 3, 256, 256).fill_(0)
            # else:
            #     mask_c = torch.FloatTensor(opt.batch_size, 3, 256, 256).fill_(0)
            # mask_c[:, :, 28:228, 28:228] = img_in
            # mask_c = mask_c.cuda()




            if i%5<opt.c_epoch:
                optimizer_c.zero_grad()
                img_c_out_raw = net_c(img_raw)
                print(img_raw.shape)
                print(img_c_out_raw.shape)
                error_c = loss_c(img_c_out_raw, target)
                error_c.backward()
                optimizer_c.step()

            # if opt.c_epoch<=i%5 and i%5<(opt.c_epoch + opt.d_epoch):
            else:
                optimizer_d.zero_grad()
                img_c_out_raw = net_c(img_raw)
                #img_c_out = img_raw.clone()
                #img_c_out[:, :, 28:228, 28:228] = img_in

                real_score = net_d(target)
                real_score = torch.squeeze(real_score)
                real_score = torch.mean(real_score)
                fake_score = net_d(img_c_out_raw)
                fake_score = torch.squeeze(fake_score)
                fake_score = torch.mean(fake_score)
                a = Variable(torch.ones(1), requires_grad=False)
                a = a.cuda()
                b = Variable(torch.zeros(1), requires_grad=False)
                b = b.cuda()

                error_a = loss_d(real_score, a)

                error_b = loss_d(fake_score, b)

                error_d = error_a +error_b
                error_d.backward()
                optimizer_d.step()
                if i%5 >= (opt.c_epoch + opt.d_epoch):
                    # optimizer_c = torch.optim.Adadelta(net_c.parameters(), rho=0.95)
                    optimizer_c.zero_grad()

                    img_c_out_raw = net_c(img_raw)
                    #img_c_out = img_raw.clone()

                    error_c = loss_c(img_c_out_raw, target)

                    # img_c_out[:, :, 28:228, 28:228] = img_in


                    fake_score = net_d(img_c_out_raw)
                    fake_score = torch.squeeze(fake_score)
                    fake_score = torch.mean(fake_score)
                    a = Variable(torch.ones(1), requires_grad=False)
                    a = a.cuda()


                    error_b = loss_d(fake_score, a)

                    error_d = error_c + opt.alpha*error_b

                    error_d.backward()

                    optimizer_c.step()

                    scheduler_c.step()


                    print('reconstruction error_c:%f, generated fake error_b:%f, total error:%f, ii:%d'  %(error_c, error_b, error_d, ii))
                    


            if (i+1)%opt.save_epoch==0:
                # print(img_c_out)
                tv.utils.save_image(img_c_out_raw.data, '%s/%s.png' %(opt.save_path, ii))
                tv.utils.save_image(img_raw.data, '%s/%s_in.jpg' %(opt.save_path, ii))
                torch.save(net_c.state_dict(), './checkpoints/net_c_%s.pth' %i)
                torch.save(net_d.state_dict(), './checkpoints/net_d_%s.pth' %i)
                optimizer_c = torch.optim.Adadelta(net_c.parameters(), rho=0.95)
                optimizer_d = torch.optim.Adadelta(net_d.parameters(), rho=0.95)
