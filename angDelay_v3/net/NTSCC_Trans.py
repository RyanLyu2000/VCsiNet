import numpy as np
import torch
import math
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste
# from compressai.ops import ste_round
from angDelay_v3.layer.analysis_transform import AnalysisTransform, Mlp
from angDelay_v3.layer.synthesis_transform import SynthesisTransform
# from csi_flatFramework.angDelay_v3.layer.hyper_encoder import Analysis_prior_net
# from csi_flatFramework.angDelay_v3.layer.hyper_decoder import Synthesis_prior_net
from angDelay_v3.layer.jscc_encoder import JSCCEncoder, SNRAdaptiveJSCCEncoder
from angDelay_v3.layer.jscc_decoder import JSCCDecoder, SNRAdaptiveJSCCDecoder
from angDelay_v3.utils.utils import BCHW2BLN, BLN2BCHW
from angDelay_v3.channel.channel import Channel
from angDelay_v3.channel.cost2100channel import Cost2100channel
from angDelay_v3.statics import evaluator
from torch.cuda.amp import autocast as autocast
import binascii

class NTC_net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        self.ga = AnalysisTransform(**config.ga_kwargs)
        self.gs = SynthesisTransform(**config.gs_kwargs)

        self.ha = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim * 3 // 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=self.embed_dim * 3 // 4, out_channels=self.embed_dim * 3 // 4, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels=self.embed_dim * 3 // 4, out_channels=self.embed_dim * 3 // 4, kernel_size=5, stride=2, padding=2),
        )
        self.hs = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.embed_dim * 3 // 4, out_channels=self.embed_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=self.embed_dim, out_channels=self.embed_dim * 3 // 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=self.embed_dim * 3 // 2, out_channels=self.embed_dim * 2, kernel_size=3, stride=1, padding=1)
        )

        self.entropy_bottleneck = EntropyBottleneck(self.embed_dim * 3 // 4)
        self.gaussian_conditional = GaussianConditional(None)

        self.channel_type = config.channel["type"]
        if self.channel_type == 'awgn':
            self.channel = Channel(config)
        else:
            self.channel = Cost2100channel(config)
        self.pass_channel = config.pass_channel

        self.fe = JSCCEncoder(**config.fe_kwargs)
        self.fd = JSCCDecoder(**config.fd_kwargs)
        if config.use_side_info:
            # hyperprior-aided decoder refinement
            embed_dim = config.fe_kwargs['embed_dim']
            self.hyprior_refinement = Mlp(embed_dim * 3, embed_dim * 6, embed_dim)

        self.eta = config.eta

        self.distortion_mse = nn.MSELoss().cuda()  # Distortion(config)
        self.distortion_nmse = evaluator

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck) )
        return aux_loss

    def forward_NTC(self, input_image):
        B, C, H, W = input_image.shape

        y = self.ga(input_image)
        z = self.ha(y)
        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset.reshape(-1, 1)
        z_hat = quantize_ste(z_tmp) + z_offset.reshape(-1, 1)

        gaussian_params = self.hs(z_hat)
        # y_hat = self.gaussian_conditional.quantize(
        #     y, "noise" if self.training else "dequantize"
        # )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        # y_likelihoods, _ = self.feature_probs_based_Gaussian(y, means_hat, scales_hat)
        y_hat = quantize_ste(y - means_hat) + means_hat
        x_hat = self.gs(y_hat)

        # 损失计算
        mse = self.distortion_mse(x_hat, input_image)
        nmse = self.distortion_nmse(x_hat, input_image)
        cs = nn.functional.cosine_similarity(x_hat, input_image).cuda()
        print(torch.mean(cs))
        # 熵值估计
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B
        return mse, bpp_y, bpp_z, x_hat, nmse

    def update(self, scale_table=None, force=False):

        SCALES_MIN = 1e-6
        SCALES_MAX = 1
        SCALES_LEVELS = 64

        def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
            return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        # updated |= self.gaussian_conditional.super().update(force=force)
        return updated

    def channel_noise(self, strings, ber):
        temp = [0 for i in range(8)]
        index = 0
        decode_strings = [i for i in strings]
        for sequences in strings:
            # a = sequences.decode('utf-16').encode('utf-8').hex()
            a = sequences.hex()
            while True:
                decode_str = ''
                for i in range(len(a)):
                    number = ord(a[i])
                    value_temp = number
                    for j in range(7, -1, -1):
                        temp[j] = value_temp % 2
                        value_temp = value_temp // 2
                    temp = np.array([v / (-0.5) + 1 for v in temp])

                    prob = np.random.rand(8)
                    justice = (prob < ber).astype(float) / (-0.5) + 1
                    decode_temp = justice * temp
                    decode_temp = ((decode_temp - 1) * (-0.5)).astype(int)
                    decode_value_temp = 0
                    for j in range(0, 8, 1):
                        decode_value_temp += decode_temp[j] * 2 ** (8 - 1 - j)
                    hex = chr(decode_value_temp)
                    decode_str += hex
                try:
                    decode_b = binascii.unhexlify(decode_str)
                    break
                except:
                    nnk = 0
            decode_strings[index] = decode_b
            index = index + 1
        return decode_strings

    def forward_NTC_test(self, input_image):
        B, C, H, W = input_image.shape

        ber = 0.005372625656130811

        y = self.ga(input_image)
        z = self.ha(y)
        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        # z_tmp = z - z_offset.reshape(-1, 1)
        # z_hat = ste_round(z_tmp) + z_offset.reshape(-1, 1)

        self.entropy_bottleneck.update()
        resprior_strings = self.entropy_bottleneck.compress(z)
        resprior_strings = self.channel_noise(resprior_strings, ber)
        z_hat = self.entropy_bottleneck.decompress(resprior_strings, z.size()[-1:])

        gaussian_params = self.hs(z_hat)
        y_rounded = self.gaussian_conditional.quantize(
            y, "dequantize"
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        self.update()
        res_indexes = self.gaussian_conditional.build_indexes(scales_hat)
        self.gaussian_conditional.update()
        res_strings = self.gaussian_conditional.compress(y_rounded, res_indexes)
        res_strings = self.channel_noise(res_strings, ber)
        y_hat = self.gaussian_conditional.decompress(res_strings, res_indexes)

        # y_likelihoods, _ = self.feature_probs_based_Gaussian(y, means_hat, scales_hat)
        # y_hat = ste_round(y - means_hat) + means_hat
        x_hat = self.gs(y_hat)

        # 损失计算
        mse = self.distortion_mse(x_hat, input_image)
        nmse = self.distortion_nmse(x_hat, input_image)
        cs = nn.functional.cosine_similarity(x_hat, input_image).cuda()

        # 熵值估计
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B
        return mse, bpp_y, bpp_z, x_hat, nmse

    def forward(self, input_image, Hu):
        B, C, H, W = input_image.shape

        # forward NTC
        y = self.ga(input_image)
        z = self.ha(y)

        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # _, z_likelihoods = self.entropy_bottleneck(z)
        # z_offset = self.entropy_bottleneck._get_medians()
        # z_tmp = z - z_offset.reshape(-1, 1)
        # z_hat = ste_round(z_tmp) + z_offset.reshape(-1, 1)

        # gaussian_params = self.hs(z_hat)
        gaussian_params = self.hs(z)
        # y_hat = self.gaussian_conditional.quantize(
        #     y, "noise" if self.training else "dequantize"
        # )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = quantize_ste(y - means_hat) + means_hat
        y_likelihoods, hy = self.feature_probs_based_Gaussian(y, means_hat, scales_hat)
        # _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        hy = torch.clamp_min(-torch.log(y_likelihoods) / math.log(2), 0)
        x_hat_ntc = self.gs(y_hat)

        # NTC-损失计算
        mse_ntc = self.distortion_mse(x_hat_ntc, input_image)
        nmse_ntc = self.distortion_nmse(x_hat_ntc, input_image)

        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B
        # bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B

        # stop backward propagation at ga
        if not self.config.joint_training:
            y = y.detach()
            y_hat = y_hat.detach()
            x_hat_ntc = x_hat_ntc.detach()

        # forward NTSCC_fefd
        s_masked, mask_BCHW, indexes = self.fe(y, y_likelihoods.detach(), hy, eta=self.eta)
        avg_pwr = torch.sum(s_masked ** 2) / mask_BCHW.sum()
        if self.channel_type == 'awgn':
            s_hat = self.channel.forward(s_masked, avg_pwr) * mask_BCHW
        else:
            s_hat = self.channel.forward(s_masked, mask_BCHW, Hu, avg_pwr) * mask_BCHW  # 需要修改信道
        # indexes
        y_ntscc_hat = self.fd(s_hat, indexes, eta=self.eta)

        if self.config.use_side_info:
            y_combine = torch.cat([BCHW2BLN(y_ntscc_hat), BCHW2BLN(means_hat), BCHW2BLN(scales_hat)], dim=-1)
            y_ntscc_hat = BLN2BCHW(BCHW2BLN(y_ntscc_hat) + self.hyprior_refinement(y_combine), H // 8, W // 8)

        # fe损失计算
        mse_y = self.distortion_mse(y_ntscc_hat, y)

        x_hat_ntscc = self.gs(y_ntscc_hat).clip(0, 1)

        # NTC-损失计算
        mse_ntscc = self.distortion_mse(x_hat_ntscc, input_image)
        nmse_ntscc = self.distortion_nmse(x_hat_ntscc, input_image)
        cs = nn.functional.cosine_similarity(x_hat_ntscc, input_image).cuda()
        print(torch.mean(cs))

        # cbr_y = mask_BCHW.sum() / (B * H * W * 3 * 2)
        cbr_y = mask_BCHW.sum() / (B * H * W * 2 * 2)  # csi只有2个通道
        return mse_ntc, bpp_y, mse_ntscc, cbr_y, x_hat_ntc, x_hat_ntscc, nmse_ntc, nmse_ntscc, mse_y


    def feature_probs_based_Gaussian(self, feature, mean, sigma):
        sigma = sigma.clamp(1e-10, 1e10) if sigma.dtype == torch.float32 else sigma.clamp(1e-10, 1e4)
        gaussian = torch.distributions.normal.Normal(mean, sigma)
        prob = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        likelihoods = torch.clamp(prob, 1e-10, 1e10)  # BCHW
        # likelihoods = -1.0 * torch.log(probs) / math.log(2.0)
        entropy = torch.clamp_min(-torch.log(likelihoods) / math.log(2), 0)  # B H W
        return likelihoods, entropy






