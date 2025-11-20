import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
from bisect import bisect
import torch.nn.functional as F
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, S, C = x.shape
    x = x.view(B, S // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, S):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    # print("windows.shape[0]", windows.shape[0], H * W)
    B = int(windows.shape[0] / (S / window_size ))
    # print(B)
    x = windows.view(B, S // window_size , window_size, -1)
    x = x.contiguous().view(B, S, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Sh
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size)
        relative_coords = coords_s[:, None] - coords_s[None, :]  # Sh * Sh
        relative_coords[:, :] += self.window_size - 1  # shift to start from 0
        relative_position_index = relative_coords
        self.register_buffer("relative_position_index", relative_position_index)        # Sh * Sh

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, add_token=True, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (N+1)x(N+1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # Sh, Sh, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Sh, Sh

        if add_token:
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # print("attention_mask: need to be maintained")
            # exit(0)
            if add_token:
                # padding mask matrix
                mask = F.pad(mask, (1, 0, 1, 0), "constant", 0)
            mask = mask.to(attn.get_device())
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class RateTokenWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        mask_r = torch.eye(window_size[0] * window_size[1], window_size[0] * window_size[1]) * 100 - 100
        self.register_buffer("mask_r", mask_r)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rate_token_windows, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # print("【Windows ATTN】 ")
        # print(x.shape)
        # print(rate_token_windows.shape)
        B_, N, C = rate_token_windows.shape
        qkv = self.qkv(x).reshape(B_, N + 2, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_r = self.qkv(rate_token_windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
        qr, _, vr = qkv_r[0], qkv_r[1], qkv_r[2]  # B H N C

        q = q * self.scale
        qr = qr * self.scale
        attn = (q @ k.transpose(-2, -1))[:, :, 2:, :]  # BxHxNx(N+2)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn[:, :, :, 2:] = attn[:, :, :, 2:] + relative_position_bias.unsqueeze(0)

        attn_r = (qr @ k[:, :, 2:, :].transpose(-2, -1)) + self.mask_r.unsqueeze(0).unsqueeze(0)  # BxHxNxN
        V = torch.cat([v, vr], dim=-2)  # BxHx(2N+2)xC

        if mask is not None:
            # padding mask matrix
            mask = F.pad(mask, (2, 0, 0, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N + 2) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N + 2)

        ATTN = torch.cat([attn, attn_r], dim=-1)  # BxHxNx(2N+2)
        ATTN = self.softmax(ATTN)
        ATTN = self.attn_drop(ATTN)

        x = (ATTN @ V).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, SNR_choice=None, eta_choice=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # build tokens
        if eta_choice is not None:
            num_eta_token = len(eta_choice)
            self.eta_choice = np.array(eta_choice)
            self.eta_token = nn.Parameter(torch.zeros(num_eta_token, 1, dim))
            trunc_normal_(self.eta_token, std=.02)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            S = self.input_resolution
            img_mask = torch.zeros((1, S, 1))  # 1 S 1
            s_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for s in s_slices:
                img_mask[:, s, :] = cnt
                cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, eta=None):
        if eta is not None:
            eta_token_id = np.searchsorted(self.eta_choice, eta.detach().cpu().numpy().reshape(-1)) - 1
            eta_token = self.eta_token[eta_token_id]
        S = self.input_resolution
        S0 = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size, input size{}x{}x{}, H:{}, W:{}".format(B, L, C, H, W)
        shortcut = x
        x = self.norm1(x)
        if S % self.window_size != 0:
            x = x.view(B, S, C)
            # pad feature maps to multiples of window size
            pad_t = 0
            pad_b = (self.window_size - S % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_t, pad_b))
            S = S + pad_b
        else:
            x = x.view(B, S, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=1)
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # add csi token
        if eta is not None:
            eta_token = eta_token.unsqueeze(1).expand(-1, B_ // B, -1, -1).reshape(B_, eta_token.shape[1], eta_token.shape[2])
            x_windows = torch.cat((eta_token, x_windows), dim=1)

        # W-MSA/SW-MSA
        if eta is not None:
            attn_windows = self.attn(x_windows,
                                     add_token=True,
                                     mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows,
                                     add_token=False,
                                     mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # remove csi token
        if eta is not None:
            attn_windows = attn_windows[:, 1:]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)
        # print(attn_windows.shape, H, W)
        shifted_x = window_reverse(attn_windows, self.window_size, S)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        # remove paddings
        if L != S:
            if pad_b > 0:
                x = x[:, :S0, :].contiguous()

        x = x.view(B, S0, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        S = self.input_resolution
        # norm1
        flops += self.dim * S
        # S-MSA
        nW = S / self.window_size
        flops += nW * self.attn.flops(self.window_size)
        # mlp
        flops += 2 * S * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * S
        return flops

    def update_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # print(self.input_resolution, self.window_size)
            S = self.input_resolution
            S = S + (self.window_size - S % self.window_size) % self.window_size

            img_mask = torch.zeros((1, S, 1))  # 1 H W 1
            s_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            cnt = 0
            for s in s_slices:
                img_mask[:, s, :] = cnt
                cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (int): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, out_dim, bias=False)
        self.norm = norm_layer(2 * dim)
        # self.proj = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        S = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)
        assert L == S, "input feature has wrong size"
        assert S % 2 == 0, f"x size (S) are not even."
        x = x.view(B, S, C)
        x0 = x[:, 0::2, :]  # B S/2 C
        x1 = x[:, 1::2, :]  # B S/2 C
        x = torch.cat([x0, x1], -1)  # B S/2 2*C
        x = self.norm(x)
        x = self.reduction(x)        # B S/2 C
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        S = self.input_resolution
        flops = S * self.dim
        flops += (S // 2) * 2 * self.dim * self.dim
        return flops


class PatchReverseMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (int): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim=0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = dim
        self.increment = nn.Linear(dim // 2, self.out_dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, S, C
        """
        S = self.input_resolution
        B, L, C = x.shape
        assert L == S, "input feature has wrong size"
        assert S % 2 == 0, f"x size (S) are not even."

        x = x.view(B, S, C//2, 2).permute(0, 1, 3, 2)
        x = x.reshape(B, 2 * S, C//2)               # B L N
        x = self.norm(x)
        x = self.increment(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        S = self.input_resolution
        flops = S * 2 * self.dim // 2
        flops += (S * 2) * self.dim // 2 * self.dim
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=2, in_chans=3, embed_dim=128, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim_1 = embed_dim // (img_size[1] // 2)         # 16 * 16 * 16 (c * (Nt // 2) * (Nc // 2)) => 256 * 16 ((c * Nc // 2), Nt // 2)
        self.embed_dim_2 = embed_dim

        self.proj = nn.Conv2d(1, self.embed_dim_1, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(self.embed_dim_2)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = x.view(B, 1, C * H, W)  # 200 * 64 * 32
        x = self.proj(x)
        x = x.permute(0, 1, 3, 2).reshape(B, self.embed_dim_2, -1)  # B C H W => B C W H => B C*W H (Batch_size channel Nt)
        x = x.transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim_1 * self.in_chans * (self.patch_size[0] * self.patch_size[1])      # projection: 2 * 32 * 32 => 16 * 16 * 16
        if self.norm is not None:
            flops += Ho * self.embed_dim_2                                                                  # normalization: 256 * 16 => 256 * 16
        return flops


class BasicLayerDec(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None,
                 SNR_choice=None, rate_choice=None, eta_choice=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, SNR_choice=SNR_choice,
                                 eta_choice=eta_choice)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, SNR=None, eta=None):
        # print(eta)
        for _, blk in enumerate(self.blocks):
            x = blk(x, eta)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            # print("blk.flops()", blk.flops())
        if self.upsample is not None:
            flops += self.upsample.flops()
            # print("upsample.flops()", self.upsample.flops())
        return flops

    def update_resolution(self, S):
        self.input_resolution = S
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = S
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = S


class BasicLayerEnc(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 SNR_choice=None, eta_choice=None):

        super().__init__()
        self.dim = dim
        if downsample is not None:
            self.input_resolution = input_resolution // 2
        else:
            self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=out_dim, input_resolution=self.input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 SNR_choice=SNR_choice, eta_choice=eta_choice)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, SNR=None, eta=None):
        if self.downsample is not None:
            x = self.downsample(x)

        for _, blk in enumerate(self.blocks):
            x = blk(x, eta)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, S):
        self.input_resolution = S
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = S // 2
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = S


class PatchDimExpansion(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.increment = nn.Linear(dim, out_dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.increment(x)
        return x