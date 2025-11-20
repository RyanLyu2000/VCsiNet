import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from torch.utils.data import Dataset, DataLoader
# from csi.NTSCC_csi.csi_distortion import csi_reshape, csi_de_reshape

class csiDatasets(Dataset):
    def __init__(self, config, train=False):
        self.train = train
        if train:
            self.data = np.load('/home/liangzijian/datasets/csiDatasets/indoor/train.npy')
        else:
            self.data = np.load('/home/liangzijian/datasets/csiDatasets/indoor/test.npy')
        self.data = torch.from_numpy(self.data)
        self.csi_channel, self.csi_height, self.csi_width = config.csi_dims
        csi_feedback_reshape = self.data.view(self.data.shape[0], self.csi_channel, self.csi_height,
                                                 self.csi_width)
        self.csi = torch.utils.data.TensorDataset(csi_feedback_reshape)
        self.len = len(self.csi)

    def __len__(self):
        return len(self.csi)       # npy文件中，batch_size预设维10

    def __getitem__(self, idx):
        return self.csi.__getitem__(idx % self.len)

def get_loader(config):
    train_dataset = csiDatasets(config, train=True)
    test_dataset = csiDatasets(config, train=False)
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=8,
                                               pin_memory=False,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader