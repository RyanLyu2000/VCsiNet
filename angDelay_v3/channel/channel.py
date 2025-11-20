import torch.nn as nn
import numpy as np
import os
import torch


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = config.channel['type']
        self.chan_param = config.channel['chan_param']
        self.device = config.device
        self.h = torch.sqrt(torch.randn(1) ** 2
                            + torch.randn(1) ** 2) / np.sqrt(2)
        # if self.chan_type == 'multipath-ofdm':
        #     self.ofdm = OFDM(self.config)
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                config.channel['type'], config.channel['chan_param']))

    def gaussian_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std):
        # fast rayleigh channel
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2) / np.sqrt(2)
        if self.config.CUDA:
            noise = noise.to(input_layer.get_device())
            h = h.to(input_layer.get_device())
        return input_layer * h + noise

    def block_rayleigh_noise_layer(self, input_layer, std):
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        if self.config.CUDA:
            noise = noise.to(input_layer.get_device())
            self.h = self.h.to(input_layer.get_device())
        out = input_layer + noise / self.h
        return out

    def update_h(self, h=None):
        if h is None:
            self.h = torch.sqrt(torch.randn(1) ** 2
                            + torch.randn(1) ** 2) / np.sqrt(2)
        else:
            self.h = h

    def block_eq_snr(self):
        sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
        new_sigma = sigma / self.h
        return torch.log10((1.0 / new_sigma ** 2) / 2) * 10

    def complex_normalize(self, x, power):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def forward(self, input, avg_pwr=False):
        # channel_tx 为 Encoder压缩后的feature，以复数符号形式发送到收端，注意
        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)

        noise = (channel_output - channel_tx).detach()
        noise.requires_grad = False
        channel_tx = channel_tx + noise
        if avg_pwr:
            return channel_tx * torch.sqrt(avg_pwr * 2)
        else:
            return channel_tx * torch.sqrt(pwr)

    def complex_forward(self, channel_in):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output

        elif self.chan_type == 'block_rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.block_rayleigh_noise_layer(channel_tx,
                                                          std=sigma)
            return chan_output

        # if self.chan_type == 3 or self.chan_type == 'multipath':
        #     channel_tx = channel_in
        #     channel_output = self.multipath_fading_noise_layer(channel_tx)
        #     return channel_output
        #
        # if self.chan_type == 4 or self.chan_type == 'multipath-ofdm':
        #     # power normalization
        #     channel_tx = channel_in
        #     channel_output = self.multipath_ofdm_fading_noise_layer(channel_tx)
        #     # print(channel_tx.shape)
        #     # print(channel_output.shape)
        #     # print(channel_tx - channel_output)
        #     return channel_output

    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx

