import math
import torch.nn as nn
import torch
from angDelay_v3.layer.analysis_transform import BasicLayer
from timm.models.layers import trunc_normal_
from angDelay_v3.layer.layers import Mlp, BasicLayerEnc
import numpy as np


class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)


class RateAdaptionEncoder(nn.Module):
    def __init__(self, channel_num, rate_choice, mode='CHW'):
        super(RateAdaptionEncoder, self).__init__()
        self.C, self.S = (channel_num, 8)
        self.rate_num = len(rate_choice)
        self.rate_choice = rate_choice
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        print("CONFIG RATE", self.rate_choice_tensor)

        self.weight = nn.Parameter(torch.zeros(self.rate_num, self.C, max(self.rate_choice)))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice)))

        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.C)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # self.weight_1 = nn.Parameter(torch.zeros(self.rate_num, self.C, self.C))
        # self.bias_1 = nn.Parameter(torch.zeros(self.rate_num, self.C))
        # self.weight_2 = nn.Parameter(torch.zeros(self.rate_num, self.C, max(self.rate_choice)))
        # self.bias_2 = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice)))
        #
        # torch.nn.init.kaiming_normal_(self.weight_1, a=math.sqrt(5))
        # torch.nn.init.kaiming_normal_(self.weight_2, a=math.sqrt(5))
        # bound_1 = 1 / math.sqrt(self.C)
        # torch.nn.init.uniform_(self.bias_1, -bound_1, bound_1)
        # bound_2 = 1 / math.sqrt(2 * self.C)
        # torch.nn.init.uniform_(self.bias_2, -bound_2, bound_2)

        # trunc_normal_(self.w, std=.02)
        mask = torch.arange(0, max(self.rate_choice)).repeat(self.S, 1)
        self.register_buffer("mask", mask)

    def forward(self, x, indexes):
        B, C, S = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        if S != self.S:
            self.update_resolution(S, x.get_device())

        # 获取MLP层参数
        w = torch.index_select(self.weight, 0, indexes).reshape(B, S, self.C, max(self.rate_choice))
        b = torch.index_select(self.bias, 0, indexes).reshape(B, S, max(self.rate_choice))

        # w_1 = torch.index_select(self.weight_1, 0, indexes).reshape(B, S, self.C, self.C)
        # b_1 = torch.index_select(self.bias_1, 0, indexes).reshape(B, S, self.C)
        # w_2 = torch.index_select(self.weight_2, 0, indexes).reshape(B, S, self.C, max(self.rate_choice))
        # b_2 = torch.index_select(self.bias_2, 0, indexes).reshape(B, S, max(self.rate_choice))

        # 获取mask信息
        mask_channelIndex = self.mask.repeat(B, 1, 1)
        mask = torch.zeros(mask_channelIndex.shape).cuda()
        rate_constraint = self.rate_choice_tensor[indexes].reshape(B, S, 1).repeat(1, 1, max(self.rate_choice))
        mask[mask_channelIndex < rate_constraint] = 1
        mask[mask_channelIndex >= rate_constraint] = 0

        # MLP层计算
        x_BLC_masked = (x_BLC + (torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b)) * mask
        # x_BLC_hidden = torch.relu(torch.matmul(x_BLC.unsqueeze(2), w_1).squeeze() + b_1)
        # x_BLC_masked = (x_BLC + (torch.matmul(x_BLC_hidden.unsqueeze(2), w_2).squeeze() + b_2)) * mask

        x_masked = x_BLC_masked.reshape(B, S, -1).permute(0, 2, 1)
        mask_BCHW = mask.reshape(B, S, -1).permute(0, 2, 1)
        return x_masked, mask_BCHW, indexes

    def update_resolution(self, S, device):
        self.S = S
        self.num_patches = S
        self.mask = torch.arange(0, max(self.rate_choice)).repeat(self.num_patches, 1)
        self.mask = self.mask.to(device)


class JSCCEncoder(nn.Module):
    def __init__(self, embed_dim=256, depths=[1, 1, 1], input_resolution=(16, 16),
                 num_heads=[8, 8, 8], window_size=(8, 16, 16),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, rate_choice=[0, 128, 256],
                 SNR_choice=None, eta_choice=None):
        super(JSCCEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList()
        self.eta_choice = eta_choice
        for i_layer in range(len(depths)):
            layer = BasicLayerEnc(dim=embed_dim, out_dim=embed_dim, input_resolution=input_resolution,
                               depth=depths[i_layer], num_heads=num_heads[i_layer],
                               window_size=window_size, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, SNR_choice=SNR_choice, eta_choice=eta_choice,
                               downsample=None)
            self.layers.append(layer)
        self.rate_adaption = RateAdaptionEncoder(embed_dim, rate_choice)
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, embed_dim))
        trunc_normal_(self.rate_token, std=.02)
        self.refine = Mlp(embed_dim * 2, embed_dim * 8, embed_dim)
        self.norm = norm_layer(embed_dim)

        # self.symbolNumber_statistics = np.array([0 for i in range(embed_dim + 1)])
        self.symbolNumber_statistics = np.array([0 for i in range(len(self.rate_choice))])

    def forward(self, x, px, hx, eta, snr=10):
        B, C, S = x.size()
        # symbol_num = torch.sum(hx, dim=1).flatten(0) * eta - 0.1
        symbol_num = (torch.sum(hx, dim=1).flatten(1) * eta - 0.1).reshape(-1)
        # symbol_num_temp = torch.ceil(symbol_num).detach().cpu().numpy()
        # for k, i in enumerate(symbol_num_temp):
        #     if int(i) <= self.embed_dim:
        #         if int(i) == 1:
        #             if symbol_num[k] < 0.1:
        #                 self.symbolNumber_statistics[0] += 1
        #             else:
        #                 self.symbolNumber_statistics[1] += 1
        #         else:
        #             self.symbolNumber_statistics[int(i)] += 1
        #     else:
        #         self.symbolNumber_statistics[self.embed_dim] += 1

        x_BLC = x.flatten(2).permute(0, 2, 1)
        px_BLC = px.flatten(2).permute(0, 2, 1)
        x_BLC = x_BLC + self.refine(torch.cat([1 - px_BLC, x_BLC], dim=-1))
        indexes = torch.searchsorted(self.rate_choice_tensor, symbol_num).clamp(0, self.rate_num - 1)  # B*H*W
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, S, C)
        x_BLC = x_BLC + rate_token
        if self.eta_choice is not None:
            for layer in self.layers:
                x_BLC = layer(x_BLC.contiguous(), SNR=snr, eta=eta)
        else:
            for layer in self.layers:
                x_BLC = layer(x_BLC.contiguous())
        x_BLC = self.norm(x_BLC)
        x_BCHW = x_BLC.reshape(B, S, C).permute(0, 2, 1)
        x_masked, mask, indexes = self.rate_adaption(x_BCHW, indexes)

        for k, i in enumerate(indexes):
            self.symbolNumber_statistics[i] += 1

        return x_masked, mask, indexes

    def update_resolution(self, S):
        self.input_resolution = S
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(S * 2)


class SNRAdaptiveJSCCEncoder(JSCCEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(self.embed_dim * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dim, self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num-1:
                outdim = self.embed_dim
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

    def get_params(self):
        params = []
        params.append(self.bm_list.parameters())
        params.append(self.sm_list.parameters())
        return params

    def forward(self, x, px, hx, eta, snr=10):
        B, C, S = x.size()
        device = x.get_device()
        # symbol_num = torch.sum(hx, dim=1).flatten(0) * eta - 0.1
        symbol_num = (torch.sum(hx, dim=1).flatten(1) * eta - 0.1).reshape(-1)
        x_BLC = x.flatten(2).permute(0, 2, 1)
        px_BLC = px.flatten(2).permute(0, 2, 1)
        x_BLC = x_BLC + self.refine(torch.cat([1 - px_BLC, x_BLC], dim=-1))
        indexes = torch.searchsorted(self.rate_choice_tensor, symbol_num).clamp(0, self.rate_num - 1)  # B*H*W
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, S, C)
        x_BLC = x_BLC + rate_token
        if self.eta_choice is not None:
            for layer in self.layers:
                x_BLC = layer(x_BLC.contiguous(), SNR=snr, eta=eta)
        else:
            for layer in self.layers:
                x_BLC = layer(x_BLC.contiguous())
        x_BLC = self.norm(x_BLC)

        # token modulation according to input snr value
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        if len(snr_cuda.shape) == 0:
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        else:
            snr_batch = snr_cuda
        input_batch = snr_batch.reshape(-1, 1)

        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x_BLC.detach())
            else:
                temp = self.sm_list[i](temp)
            bm = self.bm_list[i](input_batch).unsqueeze(1).expand(-1, S, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x_BLC = x_BLC * mod_val

        x_BCHW = x_BLC.reshape(B, S, C).permute(0, 2, 1)
        x_masked, mask, indexes = self.rate_adaption(x_BCHW, indexes)
        return x_masked, mask, indexes


# def build_model():
#     feature = torch.zeros([4, 256, 8, 16, 16])
#     analysis_prior_net = Analysis_prior_net()
#     z = analysis_prior_net(feature)
#     print(feature.size())
#     print(z.size())
#
#
# if __name__ == '__main__':
#     build_model()
