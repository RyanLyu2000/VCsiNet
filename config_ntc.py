import torch
import torch.nn as nn
from datetime import datetime

class config_ntc(object):
    """
    Shared NTC config
    """
    trainset = 'csiDataset_indoor'
    base_path = '/media/D/liangzijian/'
    data_dir = '/media/D/liangzijian/csi/%s/' % trainset
    # %s 是指在这个之后替换一个词
    csi_dims = (2, 32, 32)
    exp_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    method = 'NTC'
    M = 12
    N = 128
    seed = 1024
    cuda = True
    gpu_id = '1'
    test_only = False
    test_entropy_estimation = True
    device = 'cuda'
    norm = False
    train_lambda = 1.6
    distortion_type = 'MSE'
    optimizer_type = 'adam'
    lr = 1e-4
    aux_lr = 1e-3
    eps = 1e-08
    weight_decay = 0
    clip_max_norm = 1

    epochs = 10000
    batch_size = 256
    milestones = 200000
    checkpoint = None
    pretrained = None
    # pretrained = "/home/lvshouye/CSI/EP634_1239999.pth.tar"
    pretrained_ntc = None
    save = True
    save_every = 40000
    test_every = 10000
    print_every = 200
    wandb = False

    image_dims = [3, 32, 32]
    train_indoor_data_dir = '/media/D/lvshouye/csiDatasets/indoor'
    test_indoor_data_dir = '/media/D/lvshouye/csiDatasets/indoor'
    train_outdoor_data_dir = '/media/D/lvshouye/csiDatasets/outdoor'
    test_outdoor_data_dir = '/media/D/lvshouye/csiDatasets/outdoor'

