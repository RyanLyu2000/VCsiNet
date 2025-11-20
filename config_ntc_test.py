import torch
import torch.nn as nn
from datetime import datetime

class config_ntc(object):
    """
    Shared NTC config
    """
    exp_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    method = 'NTC'
    M = 48
    N = 128
    seed = 1024
    cuda = True
    gpu_id = '1'
    test_only = True
    test_entropy_estimation = True
    device = 'cuda'
    norm = False
    train_lambda = 0.2
    distortion_type = 'MSE'
    optimizer_type = 'adam'
    lr = 1e-4
    aux_lr = 1e-3
    eps = 1e-08
    weight_decay = 0
    clip_max_norm = 1

    epochs = 10000
    batch_size = 256
    checkpoint = None
    pretrained = None
    pretrained_ntc = None
    save = True
    save_every = 80000
    test_every = 4000
    print_every = 200
    wandb = False

    image_dims = [3, 32, 32]
    train_data_dir = '/media/D/Dataset/CIFAR10'
    test_data_dir = '/media/D/Dataset/CIFAR10'

