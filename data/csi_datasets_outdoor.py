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
        self.csi_downlink = np.sort(glob(os.path.join("/home/liangzijian/datasets/csi/csiDataset_outdoor/downlink_ifft_fft_32_32_norm", "*.npy")))
        # self.csi_downlink = np.sort(glob(os.path.join("/home/liangzijian/datasets/csi/csiOriginDataset/outdoor", "*.npy")))
        self.csi_uplink = np.sort(glob(os.path.join("/home/liangzijian/datasets/csi/csiDataset_outdoor/uplink_npy", "*.npy")))
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
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader
