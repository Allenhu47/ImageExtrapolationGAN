#coding=utf-8
import torch
from train import train

class Config(object):
    data_path = 'image'
    batch_size = 1
    num_workers = 16
    max_epoch = 300
    c_epoch = 2
    d_epoch = 1
    alpha = 0.004
    save_epoch = 50
    save_path = 'image_gen'
    use_gpu = True


if __name__=='__main__':
    opt = Config()
    train(opt)
