#coding=utf-8
import torch
from train_v1 import train

class Config(object):
    data_path = 'image/'
    batch_size = 64
    num_workers = 16
    max_epoch = 200
    c_epoch = 1
    d_epoch = 3
    alpha = 0.004
    save_epoch = 50
    save_path = 'image_gen'
    use_gpu = True


if __name__=='__main__':
    opt = Config()
    train(opt)
