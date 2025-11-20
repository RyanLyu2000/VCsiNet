import torch
import torch.nn as nn
import math

def evaluator(sparse_pred, sparse_gt):
    r""" Evaluation of decoding implemented in PyTorch Tensor
         Computes normalized mean square error (NMSE) and rho.
    """

    with torch.no_grad():
        # Basic params
        nt = 32
        nc = 32
        nc_expand = 257

        # De-centralize
        sparse_gt = sparse_gt - 0.5
        sparse_pred = sparse_pred - 0.5

        # Calculate the NMSE
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())

        # # Calculate the Rho
        # n = sparse_pred.size(0)
        # sparse_pred = sparse_pred.permute(0, 2, 3, 1)  # Move the real/imaginary dim to the last
        # zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))
        # sparse_pred = torch.cat((sparse_pred, zeros), dim=2)
        # raw_pred = torch.fft(sparse_pred, signal_ndim=1)[:, :, :125, :]
        #
        # norm_pred = raw_pred[..., 0] ** 2 + raw_pred[..., 1] ** 2
        # norm_pred = torch.sqrt(norm_pred.sum(dim=1))

        # norm_gt = raw_gt[..., 0] ** 2 + raw_gt[..., 1] ** 2
        # norm_gt = torch.sqrt(norm_gt.sum(dim=1))
        #
        # real_cross = raw_pred[..., 0] * raw_gt[..., 0] + raw_pred[..., 1] * raw_gt[..., 1]
        # real_cross = real_cross.sum(dim=1)
        # imag_cross = raw_pred[..., 0] * raw_gt[..., 1] - raw_pred[..., 1] * raw_gt[..., 0]
        # imag_cross = imag_cross.sum(dim=1)
        # norm_cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)
        #
        # rho = (norm_cross / (norm_pred * norm_gt)).mean()
        #
        # return rho, nmse

        return nmse


def ADtoSF(angDelay, Nc=1024, Nt=32):
    """
    angDelay_v1 domain to spatialFreq domain, with Nc = 1024,
    :param angDelay:        csi - angDelay_v1 domain   2 * Nt * Nc' (Nc' <= Nc)
    :param Nc:              subcarrier number
    :param Nt:              transmit antenna number at BS
    :return: spatialFreq    csi - spatialFreq domain
    """
    [batch_size, channel, Nt_, Nc_] = angDelay.shape
    if channel != 2:
        angDelay = angDelay.view(batch_size, 2, channel // 2, Nt_, Nc_)
        angDelay = torch.permute(angDelay, [0, 1, 3, 2, 4])
        angDelay = angDelay.reshape(batch_size, 2, Nt_, (channel // 2) * Nc_)
        Nc_ = (channel // 2) * Nc_

    if Nt_ < Nt:
        zeros_Nt = torch.zeros(batch_size, channel, Nt - Nt_, Nc_).cuda()
        torch.nn.init.constant_(zeros_Nt, 0.5)
        angDelay = torch.cat((angDelay, zeros_Nt), dim=2)

    if Nc_ < Nc:
        zeros_Nc = torch.zeros(batch_size, channel, Nt, Nc - Nc_).cuda()
        torch.nn.init.constant_(zeros_Nc, 0.5)
        angDelay = torch.cat((angDelay, zeros_Nc), dim=3)

    angDelay_zeroMean = angDelay - torch.tensor(0.5)                      # batch_size * 2 * Nt * Nc
    angDelay_complex = torch.squeeze(angDelay_zeroMean[:, 0, :, :]) + 1j * torch.squeeze(angDelay_zeroMean[:, 1, :, :])   # batchsize * Nt * Nc
    spatialDelay = torch.fft.ifft(angDelay_complex, n=Nt, dim=1)
    spatialDelay = torch.permute(spatialDelay, [0, 2, 1])
    spatialFreq = torch.fft.fft(spatialDelay, n=Nc, dim=1)                # Nc * Nt

    return spatialFreq

def nmse_evaluator(sparse_pred, sparse_gt, mode="process"):
    """
    calculate the nmse
    :param sparse_pred: pred
    :param sparse_gt:   groundTruth
    :param mode:        "process" ===> ( spatialFreq => AngDelay =(normalize & w/ or w/o truncate)=> AngDealyProcess
                        other ===> spatialFreq
    :return:     NMSE
    """

    with torch.no_grad():
        [batch_size, channel, Nt, Nc_] = sparse_pred.shape

        if mode == "process":
            gt = ADtoSF(sparse_gt, Nc=1024)
            pred = ADtoSF(sparse_pred, Nc=1024)

            gt = torch.cat((torch.real(gt).unsqueeze(dim=1), torch.imag(gt).unsqueeze(dim=1)), dim=1)
            pred = torch.cat((torch.real(pred).unsqueeze(dim=1), torch.imag(pred).unsqueeze(dim=1)), dim=1)

        else:
            pred = sparse_pred.view(batch_size, 2, channel // 2, Nt, Nc_)
            pred = torch.permute(pred, [0, 1, 3, 4, 2])
            pred = pred.reshape(batch_size, 2, Nt, Nc_ * (channel // 2))

            gt = sparse_gt.view(batch_size, 2, channel // 2, Nt, Nc_)
            gt = torch.permute(gt, [0, 1, 3, 4, 2])
            gt = gt.reshape(batch_size, 2, Nt, Nc_ * (channel // 2))

        power_gt = torch.sum(gt ** 2, dim=1)
        difference = gt - pred
        mse = torch.sum(difference ** 2, dim=1)
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())

        return nmse


def cosine_similarity(pred, gt, mode="process"):

    with torch.no_grad():
        [batch_size, channel, Nt, Nc_] = pred.shape

        if mode == "process":
            spatialFreq_pred = ADtoSF(pred, Nc=1024)
            spatialFreq_gt = ADtoSF(gt, Nc=1024)
        else:
            spatialFreq_pred = pred.view(batch_size, 2, channel // 2, Nt, Nc_)
            spatialFreq_pred = torch.permute(spatialFreq_pred, [0, 1, 3, 4, 2])
            spatialFreq_pred = spatialFreq_pred.reshape(batch_size, 2, Nt, Nc_ * (channel // 2))
            spatialFreq_pred = torch.squeeze(spatialFreq_pred[:, 0, :, :]) + 1j * torch.squeeze(spatialFreq_pred[:, 2, :, :])

            spatialFreq_gt = gt.view(batch_size, 2, channel // 2, Nt, Nc_)
            spatialFreq_gt = torch.permute(spatialFreq_gt, [0, 1, 3, 4, 2])
            spatialFreq_gt = spatialFreq_gt.reshape(batch_size, 2, Nt, Nc_ * (channel // 2))
            spatialFreq_gt = torch.squeeze(spatialFreq_gt[:, 0, :, :]) + 1j * torch.squeeze(spatialFreq_gt[:, 1, :, :])

        # calculate cosine similarity
        numerator = torch.abs(torch.sum(spatialFreq_pred * torch.conj(spatialFreq_gt), dim=2))
        denominator = torch.sqrt(torch.sum(torch.real(spatialFreq_pred) ** 2 + torch.imag(spatialFreq_pred) ** 2, dim=2)) \
                      * torch.sqrt(torch.sum(torch.real(spatialFreq_gt) ** 2 + torch.imag(spatialFreq_gt) ** 2, dim=2))
        similarity = torch.mean(torch.mean(numerator / denominator, dim=1), dim=0)

        return similarity


def pearson_similarity(pred, gt, mode="process"):
    with torch.no_grad():
        [batch_size, channel, Nt, Nc_] = pred.shape

        if mode == "process":
            spatialFreq_pred = ADtoSF(pred, Nc=1024)
            spatialFreq_gt = ADtoSF(gt, Nc=1024)
        else:
            spatialFreq_pred = pred.view(batch_size, 2, channel // 2, Nt, Nc_)
            spatialFreq_pred = torch.permute(spatialFreq_pred, [0, 1, 3, 4, 2])
            spatialFreq_pred = spatialFreq_pred.reshape(batch_size, 2, Nt, Nc_ * (channel // 2))
            spatialFreq_pred = torch.squeeze(spatialFreq_pred[:, 0, :, :]) + 1j * torch.squeeze(
                spatialFreq_pred[:, 2, :, :])

            spatialFreq_gt = gt.view(batch_size, 2, channel // 2, Nt, Nc_)
            spatialFreq_gt = torch.permute(spatialFreq_gt, [0, 1, 3, 4, 2])
            spatialFreq_gt = spatialFreq_gt.reshape(batch_size, 2, Nt, Nc_ * (channel // 2))
            spatialFreq_gt = torch.squeeze(spatialFreq_gt[:, 0, :, :]) + 1j * torch.squeeze(spatialFreq_gt[:, 1, :, :])

        spatialFreq_pred = spatialFreq_pred - torch.mean(spatialFreq_pred, dim=2).reshape(200, 1024, 1)
        spatialFreq_gt = spatialFreq_gt - torch.mean(spatialFreq_gt, dim=2).reshape(200, 1024, 1)

        # calculate cosine similarity
        numerator = torch.abs(torch.sum(spatialFreq_pred * torch.conj(spatialFreq_gt), dim=2))
        denominator = torch.sqrt(
            torch.sum(torch.real(spatialFreq_pred) ** 2 + torch.imag(spatialFreq_pred) ** 2, dim=2)) \
                      * torch.sqrt(torch.sum(torch.real(spatialFreq_gt) ** 2 + torch.imag(spatialFreq_gt) ** 2, dim=2))
        similarity = torch.mean(torch.mean(numerator / denominator, dim=1), dim=0)

        return similarity


class channelWise_MSE(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        self.channel_num = channel_num
        self.mse = nn.MSELoss().cuda()

    def forward(self, recon_csi, csi):
        csi = csi - 0.5
        recon_csi = recon_csi - 0.5

        channel_mse = torch.zeros(self.channel_num).cuda()
        channel_power = torch.zeros(self.channel_num).cuda()
        for i in range(self.channel_num):
            channel_mse[i] = self.mse(recon_csi[:, i, :, :], csi[:, i, :, :])
            channel_power[i] = torch.sum(csi[:, i, :, :] ** 2).detach()
        channelWise_MSE = torch.sum(channel_power * channel_mse) / torch.sum(channel_power)
        return channelWise_MSE


class spatialFreq_MSE(nn.Module):
    def __init__(self):
        super(spatialFreq_MSE, self).__init__()
        self.mse = nn.MSELoss().cuda()

    def forward(self, gt_angDelay, pred_angDelay):
        spatialFreq_pred = ADtoSF(pred_angDelay, Nc=1024)
        spatialFreq_gt = ADtoSF(gt_angDelay, Nc=1024)

        spatialFreq_gt = torch.cat((torch.real(spatialFreq_gt).unsqueeze(dim=1), torch.imag(spatialFreq_gt).unsqueeze(dim=1)), dim=1)
        spatialFreq_pred = torch.cat((torch.real(spatialFreq_pred).unsqueeze(dim=1), torch.imag(spatialFreq_pred).unsqueeze(dim=1)), dim=1)

        mse = self.mse(spatialFreq_pred, spatialFreq_gt)

        return mse


class CalThroughput(nn.Module):
    def __init__(self, batch_size, userNum):
        super(CalThroughput).__init__()
        self.batch_size = batch_size
        self.userNum = userNum
        if self.batch_size % self.userNum != 0:
            print("error: batch_size cannot fit the MU-MIMO configuration")
            exit(0)
        else:
            self.group_size = self.batch_size // self.userNum

    def forward(self, gt_angDelay, pred_angDelay, downlinkSNR):
        spatialFreq_pred = ADtoSF(pred_angDelay, Nc=1024)   # 估计的空频域CSI      batch_size * Nc * Nt
        spatialFreq_gt = ADtoSF(gt_angDelay, Nc=1024)       # 实际的空频域CSI      batch_size * Nc * Nt
        [batch_size, Nc, Nt] = spatialFreq_pred.shape

        spatialFreq_pred_forUserGroup_tmp = spatialFreq_pred.reshape(self.group_size, self.userNum, Nc, Nt)
        spatialFreq_gt_forUserGroup_tmp = spatialFreq_gt.reshape(self.group_size, self.userNum, Nc, Nt)
        spatialFreq_pred_forUserGroup = spatialFreq_pred_forUserGroup_tmp.permute(0, 2, 1, 3).reshape(self.group_size * Nc, self.userNum, Nt)     # (group_size * Nc) * userNum * Nt
        spatialFreq_gt_forUserGroup = spatialFreq_gt_forUserGroup_tmp.permute(0, 2, 1, 3).reshape(self.group_size * Nc, self.userNum, Nt)

        # 计算预编码矩阵和等效信道响应
        precoding_matrix = torch.linalg.pinv(spatialFreq_pred_forUserGroup, hermitian=True)                                 # (self.group_size * Nc) * Nt * self.userNum
        scaling_factor = torch.abs(torch.sum(precoding_matrix, dim=2)) ** 2
        scaling_factor = torch.mean(scaling_factor.reshape(self.group_size, Nc, Nt), dim=[1, 2]).reshape(-1, 1).repeat(1, Nc).reshape(self.group_size * Nc)
        precoding_matrix = precoding_matrix / scaling_factor                                                                # 预编码矩阵的功率归一化
        effectiveChannel = spatialFreq_gt_forUserGroup @ precoding_matrix                                                   # (self.group_size * Nc) * self.userNum * self.userNum

        # 计算噪声功率
        power_ofdm = torch.sum(torch.abs(spatialFreq_gt_forUserGroup_tmp) ** 2, dim=3)                                                  # self.group_size * self.userNum(r) * Nc
        power_mean = torch.mean(power_ofdm, dim=2).squeeze()                                                                            # self.group_size * self.userNum(r)
        downlinkSNR = downlinkSNR.reshape(-1, 1).expand(self.batch_size, 1).reshape(self.group_size, self.userNum)                      # self.group_size * self.userNum(r)
        sigma2 = torch.sqrt(power_mean / (10 ** (downlinkSNR / 10))).reshape(-1, 1)                                                     # self.group_size * self.userNum(r)
        sigma2 = sigma2.reshape(self.group_size, 1, self.userNum).repeat(1, Nc, 1).reshape(self.group_size * Nc, self.userNum)          # (self.group_size * Nc) * self.userNum(r)

        # 计算MMSE矩阵
        throughput_MU = torch.zeros(1).cuda()
        distance_MU = torch.zeros(1).cuda()
        for u in range(self.userNum):
            # 计算信干噪比
            effeChan_forU = effectiveChannel[:, u, :]
            effeChan_H_forU = effeChan_forU.permute(0, 2, 1).conj()
            sigma2_forU = sigma2[:, u].reshape(-1, 1, 1).reshape(1, 1, 2)
            I = torch.ones(sigma2_forU.shape[0], self.userNum, self.userNum).cuda()
            W_mmse = torch.linalg.inv(effeChan_H_forU @ effeChan_forU + sigma2 @ I) @ effeChan_H_forU           # (self.batch_size * Nc) * self.userNum * 1
            W_mmse_u = W_mmse[:, u, :].squeeze()
            effeChan_forU_u = effeChan_forU[:, :, u].reshape()

            numerator = torch.abs(W_mmse_u * effeChan_forU_u) ** 2
            denominator_core = torch.zeros(numerator.shape[0]).cuda()
            for v in range(self.userNum - 1):
                effeChan_forU_v = effeChan_forU[:, :, v].reshape()
                denominator_core += effeChan_forU_u * effeChan_forU_v.conj()
            denominator = W_mmse_u * (denominator_core + sigma2[:, u].reshape()) * W_mmse_u.conj()
            SINR_u = numerator / denominator

            throughput = torch.mean(torch.log2(1 + SINR_u))
            throughput_MU += throughput

            # 计算上限（信噪比）
            ideal_numerator_u = power_ofdm[:, u, :].squeeze().reshape(self.group_size * Nc)
            sigma2_u = sigma2[:, u].squeeze()
            SNR_u = ideal_numerator_u / sigma2_u
            ideal_throughput = torch.mean(torch.log2(1 + SNR_u))
            distance = ideal_throughput - throughput
            distance_MU += distance

        return distance_MU, throughput_MU






