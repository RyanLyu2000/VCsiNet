import math
import time

import numpy as np
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from timm.models.layers import trunc_normal_
from net.utils import conv, deconv, update_registered_buffers, quantize_ste, DEMUX, MUX
from compressai.layers import GDN

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class NTC(nn.Module):
    r"""Nonlinear transform coding using checkerboard entropy model.
    The backbone of ga&gs are Neighborhood Attention Transformers or ELIC.
    """

    def __init__(self, N=256, M=320):
        super().__init__()

        self.num_iters = 1
        self.M = M
        self.g_a = nn.Sequential(
            conv(2, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M, stride=1),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, stride=1),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 2),
        )
        self.mean = nn.Parameter(torch.zeros(M, 8, 8))
        self.scale = nn.Parameter(torch.ones(M, 8, 8))
        self.H = nn.Parameter(torch.ones(M, 64, 64))
        trunc_normal_(self.H, std=.02)
        self.mask = torch.tril(torch.ones(64, 64).cuda()) - torch.eye(64).cuda()


        self.gaussian_conditional = GaussianConditional(None)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def forward(self, x):
        y = self.g_a(x)
        B, C, h, w = y.shape
        y_reshape = y.reshape(B*C, h*w, 1)
        H_tril = self.H * self.mask
        y_ind = torch.bmm(H_tril.unsqueeze(0).repeat(B, 1, 1, 1).reshape(B*C, 64, 64), y_reshape)
        w_res = y_reshape - y_ind
        w_res_hat, w_res_likelihoods = self.gaussian_conditional(w_res.reshape(B, C, h, w), self.scale.unsqueeze(0).repeat(B, 1, 1, 1), self.mean.unsqueeze(0).repeat(B, 1, 1, 1))
        H_inverse = torch.inverse(torch.eye(64).unsqueeze(0).cuda()-H_tril)
        # print(H_tril)
        # print(H_inverse)
        y_hat = torch.matmul(H_inverse.unsqueeze(0).repeat(B, 1, 1, 1).reshape(B*C, 64, 64), w_res_hat.reshape(B*C, h*w, 1)).reshape(B, C, h, w)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"w_res": w_res_likelihoods},
            "y":y,
            "y_hat":y_hat,
            "w_res":w_res.reshape(B, C, h, w),
            "w_res_hat":w_res_hat,
        }



    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        print(len(y_strings), len(y_strings[0]), len(z_strings), len(z_strings[0]))
        return {"y": y,
                "scales_hat": scales_hat,
                "means_hat": means_hat,
                "strings": [y_strings, z_strings],
                "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "y_hat": y_hat,
            "x_hat": x_hat}


