import math
import torch.nn as nn
import torch
from angDelay_v3.layer.layers import BasicLayerDec
from timm.models.layers import trunc_normal_
import numpy as np
from angDelay_v3.layer.jscc_encoder_etaAttention import AdaptiveModulator


class RateAdaptionDecoder(nn.Module):
    def __init__(self, channel_num, rate_choice, mode='CHW'):
        super(RateAdaptionDecoder, self).__init__()
        self.C = channel_num
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)

        self.weight = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice), self.C))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, self.C))

        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.rate_num)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # self.weight_1 = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice), self.C))
        # self.bias_1 = nn.Parameter(torch.zeros(self.rate_num, self.C))
        # self.weight_2 = nn.Parameter(torch.zeros(self.rate_num,self.C, self.C))
        # self.bias_2 = nn.Parameter(torch.zeros(self.rate_num, self.C))
        #
        # torch.nn.init.kaiming_normal_(self.weight_1, a=math.sqrt(5))
        # torch.nn.init.kaiming_normal_(self.weight_2, a=math.sqrt(5))
        # bound_1 = 1 / math.sqrt(self.rate_num)
        # torch.nn.init.uniform_(self.bias_1, -bound_1, bound_1)
        # bound_2 = 1 / math.sqrt(2 * self.C)
        # torch.nn.init.uniform_(self.bias_2, -bound_2, bound_2)

        # trunc_normal_(self.weight_bias, std=.02)

    def forward(self, x, indexes):
        B, _, S = x.size()
        x_BLC_masked = x.flatten(2).permute(0, 2, 1)
        # 获取MLP参数
        w = torch.index_select(self.weight, 0, indexes).reshape(B, S, max(self.rate_choice), self.C)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, S, self.C)

        # w_1 = torch.index_select(self.weight_1, 0, indexes).reshape(B, S, max(self.rate_choice), self.C)
        # b_1 = torch.index_select(self.bias_1, 0, indexes).reshape(B, S, self.C)
        # w_2 = torch.index_select(self.weight_2, 0, indexes).reshape(B, S, self.C, self.C)
        # b_2 = torch.index_select(self.bias_2, 0, indexes).reshape(B, S, self.C)
        # print(w.dtype)
        # print(b.dtype)
        # print(x_BLC.dtype)
        # MLP计算
        x_BLC = x_BLC_masked + (torch.matmul(x_BLC_masked.unsqueeze(2), w).squeeze() + b)  # BLN
        # x_BLC_hidden = torch.relu(torch.matmul(x_BLC_masked.unsqueeze(2), w_1).squeeze() + b_1)
        # x_BLC = x_BLC_masked + torch.matmul(x_BLC_hidden.unsqueeze(2), w_2).squeeze() + b_2

        out = x_BLC.reshape(B, S, -1).permute(0, 2, 1)
        return out


class JSCCDecoder(nn.Module):
    def __init__(self, embed_dim=256, depths=[1, 1, 1],
                 input_resolution=(16, 16),
                 num_heads=[8, 8, 8], window_size=(8, 16, 16),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 SNR_choice=None, eta_choice=None, rate_choice=[0, 128, 256]):
        super(JSCCDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.eta_choice = eta_choice
        for i_layer in range(len(depths)):
            layer = BasicLayerDec(dim=embed_dim, out_dim=embed_dim, input_resolution=input_resolution,
                                  depth=depths[i_layer], num_heads=num_heads[i_layer],
                                  window_size=window_size, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  SNR_choice=SNR_choice, eta_choice=eta_choice,
                                  upsample=None)
            self.layers.append(layer)
        self.embed_dim = embed_dim
        self.rate_adaption = RateAdaptionDecoder(embed_dim, rate_choice)
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, embed_dim))
        trunc_normal_(self.rate_token, std=.02)

    def forward(self, x, indexes, eta, snr=10):
        B, _, S = x.size()
        x = self.rate_adaption(x, indexes)

        x_BLC = x.flatten(2).permute(0, 2, 1)
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, S, self.embed_dim)

        x_BLC = x_BLC + rate_token
        if self.eta_choice is not None:
            for layer in self.layers:
                x_BLC = layer(x_BLC.contiguous(), SNR=snr, eta=eta)
        else:
            for layer in self.layers:
                x_BLC = layer(x_BLC.contiguous())
        x_BCHW = x_BLC.reshape(B, S, self.embed_dim).permute(0, 2, 1)
        return x_BCHW

    def update_resolution(self, S):
        self.input_resolution = S
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(S)


class SNRAdaptiveJSCCDecoder(JSCCDecoder):
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

    def forward(self, x, indexes, eta, snr=10):
        B, _, S = x.size()
        device = x.get_device()
        x = self.rate_adaption(x, indexes)
        x_BLC = x.flatten(2).permute(0, 2, 1)

        # token modulation according to input snr value
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        if len(snr_cuda.shape) == 0:
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        else:
            snr_batch = snr_cuda
        input_batch = snr_batch.reshape(-1, 1)

        for i in range(self.layer_num):
            if i == 0:
                temp_1 = self.sm_list[i](x_BLC.detach())
            else:
                temp_1 = self.sm_list[i](temp) + temp
            bm = self.bm_list[i](input_batch).unsqueeze(1).reshape(-1, S, self.hidden_dim)
            temp = temp_1 * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x_BLC = x_BLC * mod_val

        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, S, self.embed_dim)

        x_BLC = x_BLC + rate_token
        if self.eta_choice is not None:
            for layer in self.layers:
                x_BLC = layer(x_BLC.contiguous(), SNR=snr, eta=eta)
        else:
            for layer in self.layers:
                x_BLC = layer(x_BLC.contiguous())
        x_BCHW = x_BLC.reshape(B, S, self.embed_dim).permute(0, 2, 1)
        return x_BCHW


# def build_model():
#     feature = torch.zeros([4, 256, 8, 4, 4])
#     analysis_prior_net = Synthesis_prior_net()
#     z = analysis_prior_net(feature)
#
#     print(feature.size())
#     print(z.size())
#
#
# if __name__ == '__main__':
#     build_model()
