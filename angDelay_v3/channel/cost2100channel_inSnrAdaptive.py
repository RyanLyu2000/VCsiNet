import torch.nn as nn
import numpy as np
import os
import torch
import math

class Cost2100channel(nn.Module):
    """
        using cost2100 uplink fading channel to generate a uplink transmission link
        receive_signal y = hx + n
        MRC receiving  x^ = h*h/||h||^2 * x + h*n / ||h||^2
        SNR = sigma||h|| * E[x] / noise^2
    """
    def __init__(self, config):
        super(Cost2100channel, self).__init__()
        self.config = config
        self.chan_type = config.channel['type']
        self.chan_param = config.channel['chan_param']
        self.device = config.device

        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                config.channel['type'], config.channel['chan_param']))

    def forward(self, input, mask_BCWH, uplinkChannel, avg_pwr=None, snr=10):
        # channel_tx 为 Encoder压缩后的feature，以复数符号形式发送到收端，注意
        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)

        snr = torch.tensor(snr, dtype=torch.float32).cuda()
        if len(snr.shape) == 0:
            snr_batch = snr.unsqueeze(0).expand(channel_tx.shape[0], -1)
        else:
            snr_batch = snr

        channel_tx_channelLast = channel_tx.permute(0, 2, 1)
        input_shape = channel_tx_channelLast.shape
        channel_in = channel_tx_channelLast.reshape(input_shape[0], -1)
        L = channel_in.shape[1]
        channel_in = channel_in[:, 0:L:2] + channel_in[:, 1:L:2] * 1j

        mask_shape = mask_BCWH.shape
        mask_channel = mask_BCWH.reshape(mask_shape[0], -1)
        # L_mask = mask_channel.shape[1]
        # mask_channel = mask_channel[:, 0:L_mask:2] + mask_channel[:, 1:L_mask:2] * 1j

        channel_out = self.cost2100_forward(channel_in, uplinkChannel, snr=snr_batch)
        # channel_out = self.awgn_forward(channel_in)

        channel_out_mat = torch.zeros((channel_out.shape[0], 2 * channel_out.shape[1])).cuda()
        channel_out_mat[:, 0:L:2] = torch.real(channel_out)
        channel_out_mat[:, 1:L:2] = torch.imag(channel_out)
        # channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_out_channelLast = channel_out_mat.reshape(input_shape)
        channel_out = channel_out_channelLast.permute(0, 2, 1)

        channel_output_mat = channel_out * mask_BCWH             # 避免无数值RE上有噪声

        noise = (channel_output_mat - channel_tx).detach()
        noise.requires_grad = False
        channel_rx = channel_tx + noise

        patches_snr = 10 * torch.log10((torch.sum(torch.abs(channel_tx) ** 2, dim=1) + torch.tensor(1e-10)) / (torch.sum(torch.abs(noise) ** 2, dim=1) + torch.tensor(1e-10)))

        if avg_pwr:
            return channel_rx * torch.sqrt(avg_pwr * 2), patches_snr
        else:
            return channel_rx * torch.sqrt(pwr), patches_snr

    def cost2100_forward(self, channel_in, uplinkChannel, snr):
        [batch_size, N] = channel_in.shape
        [batch_size, Nc, Nt] = uplinkChannel.shape

        batch_size_forChannel = batch_size
        N_forChannel = N
        if N > Nc:
            channel_in = channel_in.reshape(batch_size, 2, N // 2)
            channel_in = channel_in.permute(1, 0, 2)
            channel_in = channel_in.reshape(2 * batch_size, N // 2)
            uplinkChannel = uplinkChannel.repeat(2, 1, 1)

            snr = snr.repeat(2, 1)                                      # snr也进行相同的复制

            batch_size_forChannel = 2 * batch_size
            N_forChannel = N // 2
            # print("符号数过多，请提高压缩率")
            # exit(0)

        snr = snr.reshape(-1)

        # cost2100 信道传输
        # size(uplinkChannel) = [batch_size, Nc, Nt, 1]
        uplinkChannel = torch.squeeze(uplinkChannel)
        uplinkChannel = uplinkChannel[:, 0:N_forChannel, :]                 # 与csi压缩后的符号数保持一致          [batch_size, N, Nt]
        channel_in = channel_in.reshape(batch_size_forChannel, N_forChannel, 1)
        channel_tx = channel_in.repeat(1, 1, Nt)

        # # 根据数值选择信道RE（比较麻烦）
        # mask_channel = mask_channel.reshape(batch_size, Nc, 1)
        # mask_channel = mask_channel.repeat(1, 1, Nt)
        # uplinkChannel = uplinkChannel * mask_channel

        uplinkChannel_power = torch.sum(torch.real(uplinkChannel) ** 2 + torch.imag(uplinkChannel) ** 2, dim=2)

        power_antenna = torch.mean(torch.real(uplinkChannel) ** 2 + torch.imag(uplinkChannel) ** 2, dim=2)  # calculate SNR in single antenna
        mean_power = torch.mean(power_antenna, dim=1)                                                       # [batch_size, 1]
        sigma = torch.sqrt(mean_power / (2 * 10 ** (snr / 10))).reshape(-1, 1, 1)                           # [batch_size, 1]
        sigma = sigma.repeat(1, N_forChannel, Nt)
        noise_real = torch.normal(mean=0, std=1, size=uplinkChannel.shape).cuda() * sigma
        noise_imag = torch.normal(mean=0, std=1, size=uplinkChannel.shape).cuda() * sigma
        noise = noise_real + 1j * noise_imag

        channel_rx = uplinkChannel * channel_tx + noise

        # MRC接收
        uplinkChannel_conjugate = torch.real(uplinkChannel) - 1j * torch.imag(uplinkChannel)
        # MRC_rx = uplinkChannel_conjugate * channel_rx
        # MRC_sum_rx = torch.sum(MRC_rx, dim=2)
        receive_rx = torch.sum(uplinkChannel_conjugate * channel_rx, dim=2) / uplinkChannel_power

        if N > Nc:
            receive_rx = receive_rx.reshape(2, batch_size, N // 2)
            receive_rx = receive_rx.permute(1, 0, 2)
            receive_rx = receive_rx.reshape(batch_size, N)

        return receive_rx

    def awgn_forward(self, channel_in):
        [batch_size, N] = channel_in.shape

        sigma = math.sqrt(1 / (2 * 10 ** (self.chan_param / 10)))    # [batch_size, 1]
        noise_real = torch.normal(mean=0, std=1, size=channel_in.shape).cuda() * sigma
        noise_imag = torch.normal(mean=0, std=1, size=channel_in.shape).cuda() * sigma
        noise = noise_real + 1j * noise_imag

        channel_rx = channel_in + noise

        return channel_rx
