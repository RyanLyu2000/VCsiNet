import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime
import os

def plot_csi(config, csi, mode="origin"):
    """
    将csi矩阵（实部与虚部）另存为png与eps
    :param csi:     CSI矩阵    numpy格式
    :return：       png与eps的输出（不返回）  origin_real.png   origin_imag.png   recons_real.png    recons_imag.png
    """
    csi = np.abs(csi - 0.5)

    for i in range(2):
        plt.imshow((np.max(np.max(csi[i])) - csi[i]))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        if mode == "origin":
            if i == 0:
                plt.savefig(config.workdir + "/origin_real.eps", format='eps', bbox_inches='tight', pad_inches = -0.05)
                plt.savefig(config.workdir + "/origin_real.png", format='png', bbox_inches='tight', pad_inches = -0.05)
            else:
                plt.savefig(config.workdir + "/origin_imag.eps", format='eps', bbox_inches='tight', pad_inches = -0.05)
                plt.savefig(config.workdir + "/origin_imag.png", format='png', bbox_inches='tight', pad_inches = -0.05)
        else:
            if i == 0:
                plt.savefig(config.workdir + "/recons_real.eps", format='eps', bbox_inches='tight', pad_inches = -0.05)
                plt.savefig(config.workdir + "/recons_real.png", format='png', bbox_inches='tight', pad_inches = -0.05)
            else:
                plt.savefig(config.workdir + "/recons_imag.eps", format='eps', bbox_inches='tight', pad_inches = -0.05)
                plt.savefig(config.workdir + "/recons_imag.png", format='png', bbox_inches='tight', pad_inches = -0.05)
        # print("pause")


def plot_csi_uniform(config, csi, max_value, mode="origin"):
    """
    将csi矩阵（实部与虚部）另存为png与eps
    :param csi:     CSI矩阵    numpy格式
    :return：       png与eps的输出（不返回）  origin_real.png   origin_imag.png   recons_real.png    recons_imag.png
    """
    csi = np.abs(csi - 0.5)

    for i in range(2):
        plt.imshow((max_value[i] - csi[i]))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        if mode == "origin":
            if i == 0:
                plt.savefig(config.workdir + "/origin_real.eps", format='eps', bbox_inches='tight', pad_inches = -0.05)
                plt.savefig(config.workdir + "/origin_real.png", format='png', bbox_inches='tight', pad_inches = -0.05)
            else:
                plt.savefig(config.workdir + "/origin_imag.eps", format='eps', bbox_inches='tight', pad_inches = -0.05)
                plt.savefig(config.workdir + "/origin_imag.png", format='png', bbox_inches='tight', pad_inches = -0.05)
        else:
            if i == 0:
                plt.savefig(config.workdir + "/recons_real.eps", format='eps', bbox_inches='tight', pad_inches = -0.05)
                plt.savefig(config.workdir + "/recons_real.png", format='png', bbox_inches='tight', pad_inches = -0.05)
            else:
                plt.savefig(config.workdir + "/recons_imag.eps", format='eps', bbox_inches='tight', pad_inches = -0.05)
                plt.savefig(config.workdir + "/recons_imag.png", format='png', bbox_inches='tight', pad_inches = -0.05)


origin1 = np.load("/media/D/liangzijian/csi_flatFramework/angDelay_v3/test_lambda1e4/2022-10-25 23:35:37/sourceCSI.npy")
recover_nmse = np.load("/media/D/liangzijian/csi_flatFramework/angDelay_v3/test_lambda1e4/2022-10-25 23:35:37/renCSI_SE.npy")
origin2 = np.load("/media/D/liangzijian/csi_flatFramework/angDelay_v3_spectralEfficiency/test_lambda2/2022-10-25 23:18:38/sourceCSI.npy")
recover_se = np.load("/media/D/liangzijian/csi_flatFramework/angDelay_v3_spectralEfficiency/test_lambda2/2022-10-25 23:18:38/renCSI_SE.npy")
print(origin1[1].shape)
print(origin2.shape)


class config1:
    filename = datetime.datetime.now().__str__()[:-7]
    workdir = "/media/D/lvshouye/CSI/csi_flatFramework/angDelay_v3_spectralEfficiency/test_lambda2/{}".format(filename)
    os.makedirs(workdir)

plot_csi(config1, origin1[1], mode="origin")
plot_csi(config1, recover_nmse[1], mode="reconstruct")

class config2:
    filename = datetime.datetime.now().__str__()[:-7]
    workdir = "/media/D/lvshouye/CSI/csi_flatFramework/angDelay_v3_spectralEfficiency/test_lambda2/{}".format(filename)
    os.makedirs(workdir)

origin1 = torch.from_numpy(origin1)
recover_nmse = torch.from_numpy(recover_nmse)
print(origin1[1].shape)
CS = F.cosine_similarity(origin1, recover_nmse, dim=1)
GCS = torch.mean(CS)
print(CS.shape)
print(GCS)

#plot_csi(config2, origin1[1],  mode="origin")
#plot_csi(config2, recover_se[1], mode="reconstruct")
#print("pause")

