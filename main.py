import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
# from csi.NTSCC_csi.csi_distortion import csi_reshape, csi_de_reshape

class csiDatasets(Dataset):
    def __init__(self, config, train=False):
        self.train = train
        if train:
            self.csi = np.load('/home/liangzijian/datasets/csiDatasets/indoor/train.npy')
            print(self.csi.shape)
        else:
            self.csi = np.load('/home/liangzijian/datasets/csiDatasets/indoor/test.npy')
        self.len = len(self.csi)
        '''
        if train:
            self.csi_downlink = self.csi_downlink[0:500]        # training set
            self.csi_uplink = self.csi_uplink[0:500]
        else:
            # self.csi_downlink = self.csi_downlink[650:750]      # testing set
            # self.csi_uplink = self.csi_uplink[650:750]
            self.csi_downlink = self.csi_downlink[500:600]  # testing set
            self.csi_uplink = self.csi_uplink[500:600]
        '''
        self.csi_channel, self.csi_height, self.csi_width = config.csi_dims

    def __len__(self):
        return len(self.csi)       # npy文件中，batch_size预设维10

    def __getitem__(self, idx):
        csi_feedback = torch.tensor(self.csi[idx], dtype=torch.float32)  # 1 * 200 * 2048
        csi_real = torch.real(csi_feedback).float()
        csi_imag = torch.imag(csi_feedback).float()
        csi_feedback = csi_real + 1j * csi_imag
        csi_feedback_reshape = csi_feedback.view(csi_feedback.shape[0], self.csi_channel, self.csi_height,
                                                 self.csi_width)  # 200 * 2 * 32 * 32
        return csi_feedback

def get_loader(config):
    train_dataset = csiDatasets(config, train=True)
    test_dataset = csiDatasets(config, train=False)
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=config.batch_size,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader

class config1:
    seed = 1024
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    trainset = 'csiDataset_indoor'
    base_path = '/media/D/liangzijian/'
    data_dir = '/media/D/liangzijian/csi/%s/' % trainset

    use_side_info = True
    train_lambda = [1e3]                      # true_lambda = 1024

    # logger
    print_step = 50
    plot_step = 1000
    filename = datetime.now().__str__()[:-7]
    workdir = '/media/D/liangzijian/csi_flatFramework/angDelay_v3/history_indoors/NTC/lambda_1e3/{}'.format(filename)
    os.makedirs(workdir)
    log = []
    epoch_log = []
    for i in train_lambda:
        log.append((workdir + '/Log_lambda%06d.log' % i))
        epoch_log.append((workdir + '/epochLog_lambda%06d.csv' % i))
    samples = workdir + '/samples'
    models = workdir + '/models'
    os.makedirs(samples)
    os.makedirs(models)
    logger = None

    distortion_metric = 'MSE'

    embed_dim = 128

    # NTC training details
    # csi_dims = (2, 1024, 32)
    # csi_dims = (2, 128, 256)
    csi_dims = (2, 32, 32)
    normalize = False
    lr = {
        "base": 0.0001,        # 0.0001,
        "decay": 0.1,
        "decay_interval": 920000
    }

    warmup_step = 1000
    tot_step = 2500000
    tot_epoch = 1000
    save_model_freq = 40000
    test_step = 500000
    batch_size = 1                  # 1 * 200
    miniBatch_size = 100
    miniBatch_num = 200 // miniBatch_size

    eta = 0.2
    pass_channel = True
    channel = {"type": 'awgn', 'chan_param': 10}
    # multiple_rate = [16, 32, 48, 64, 80, 96, 102, 118, 134, 160, 186, 192, 208, 224, 240, 256]
    multiple_rate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256]
    # multiple_rate = []

    ga_kwargs = dict(
        img_size=(csi_dims[1], csi_dims[2]), in_chans=2,
        embed_dims=[embed_dim, embed_dim, embed_dim], depths=[1, 1, 1], num_heads=[8, 8, 8],
        window_size=4, mlp_ratio=8., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )

    gs_kwargs = dict(
        img_size=(csi_dims[1], csi_dims[2]), out_chans=2,
        embed_dims=[embed_dim, embed_dim, embed_dim], depths=[1, 1, 1], num_heads=[8, 8, 8],
        window_size=4, mlp_ratio=8., norm_layer=nn.LayerNorm, patch_norm=True
    )

    # ha_kwargs = dict(
    #     input_resolution=(csi_dims[1] // 8), out_dim=embed_dim,
    #     embed_dim=embed_dim, depths=[1, 1, 1], num_heads=[8, 8, 8],
    #     window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #     drop_rate=0., attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm
    # )
    #
    # hs_kwargs = dict(
    #     input_resolution=(csi_dims[1] // 8), out_dim=embed_dim,
    #     embed_dim=embed_dim, depths=[1, 1, 1], num_heads=[8, 8, 8],
    #     window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #     drop_rate=0., attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm
    # )

    fe_kwargs = dict(
        input_resolution=(csi_dims[1] // 8),
        embed_dim=embed_dim, depths=[4], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=multiple_rate
    )

    fd_kwargs = dict(
        input_resolution=(csi_dims[1] // 8),
        embed_dim=embed_dim, depths=[4], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=multiple_rate
    )

train_loader, test_loader = get_loader(config1)
print(type(train_loader))

'''
class csiDatasets(Dataset):
    def __init__(self, config, train=False):
        self.train = train
        self.csi_downlink = np.sort(glob(os.path.join("/home/liangzijian/datasets/csi/csiDataset_indoor/downlink_ifft_fft_32_32_norm", "*.npy")))
        # self.csi_downlink = np.sort(glob(os.path.join("/media/D/liangzijian/csi/csiOriginDataset/indoor", "*.npy")))
        self.csi_uplink = np.sort(glob(os.path.join("/home/liangzijian/datasets/csi/csiDataset_indoor/uplink_npy", "*.npy")))
        self.len = len(self.csi_downlink)
        if train:
            self.csi_downlink = self.csi_downlink[0:500]        # training set
            self.csi_uplink = self.csi_uplink[0:500]
        else:
            # self.csi_downlink = self.csi_downlink[650:750]      # testing set
            # self.csi_uplink = self.csi_uplink[650:750]
            self.csi_downlink = self.csi_downlink[500:600]  # testing set
            self.csi_uplink = self.csi_uplink[500:600]

        self.csi_channel, self.csi_height, self.csi_width = config.csi_dims

    def __len__(self):
        print(self.csi_downlink)
        return len(self.csi_downlink)       # npy文件中，batch_size预设维10

    def __getitem__(self, idx):
        csi_feedback = torch.tensor(np.load(self.csi_downlink[idx]), dtype=torch.float32)       # 1 * 200 * 2048
        csi_feedback_reshape = csi_feedback.view(csi_feedback.shape[0], self.csi_channel, self.csi_height, self.csi_width)      # 200 * 2 * 32 * 32

        Hu_uplink = torch.tensor(np.load(self.csi_uplink[idx]), dtype=torch.complex64)

        Hu_uplink_real = torch.real(Hu_uplink).float()
        Hu_uplink_imag = torch.imag(Hu_uplink).float()
        Hu_uplink = Hu_uplink_real + 1j * Hu_uplink_imag
        return [csi_feedback_reshape, Hu_uplink]
        # return csi_feedback_reshape

def get_loader(config):
    train_dataset = csiDatasets(config, train=True)
    test_dataset = csiDatasets(config, train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=config.batch_size,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader

'''