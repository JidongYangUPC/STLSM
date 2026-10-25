#
# 3-input-1_1-mse-ssim-1e-lsm-efficient
# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.
#
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """
#
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#
#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)
#
#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
#
#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., window_size=7):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # EffAttention的降维策略
        self.reduce = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 从WindowAttention引入相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # 相对位置索引计算
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords, coords]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # 降维
        x_reduced = self.reduce(x)
        B, N, C = x_reduced.shape

        # QKV计算
        qkv = self.qkv(x_reduced).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # EffAttention的分块计算
        chunk_size = max(1, N // 4)  # 动态分块
        q_chunks = torch.split(q, chunk_size, dim=-2)
        k_chunks = torch.split(k, chunk_size, dim=-2)
        v_chunks = torch.split(v, chunk_size, dim=-2)

        outputs = []
        for i, (q_chunk, k_chunk, v_chunk) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
            # 注意力计算
            attn = (q_chunk @ k_chunk.transpose(-2, -1)) * self.scale

            # 添加相对位置偏置（关键改进！）
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, N)
            chunk_relative_bias = self._get_chunk_relative_bias(chunk_start, chunk_end, N)
            attn = attn + chunk_relative_bias

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            chunk_output = (attn @ v_chunk).transpose(1, 2)
            outputs.append(chunk_output)

        # 合并结果
        x = torch.cat(outputs, dim=1)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _get_chunk_relative_bias(self, start, end, total_len):
        # 获取当前chunk的相对位置偏置
        indices = self.relative_position_index[start:end, start:end]
        relative_bias = self.relative_position_bias_table[indices.reshape(-1)]
        relative_bias = relative_bias.reshape(end - start, end - start, -1).permute(2, 0, 1)
        return relative_bias.unsqueeze(0)

class EdgeAwareAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 原有的相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 边缘敏感的位置偏置（新增）
        self.edge_bias = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 获取相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # 原有的QKV和投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.edge_bias, std=.02)  # 初始化边缘偏置
        self.softmax = nn.Softmax(dim=-1)

        # 边缘检测卷积核
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                     dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                     dtype=torch.float32).view(1, 1, 3, 3))

    def compute_edge_features(self, x):
        """计算边缘特征"""
        B, H, W, C = x.shape

        # 处理任意通道数的输入
        if C == 1:
            # 单通道直接使用
            gray = x.squeeze(-1)
        elif C == 3:
            # RGB转换为灰度
            gray = 0.299 * x[:, :, :, 0] + 0.587 * x[:, :, :, 1] + 0.114 * x[:, :, :, 2]
        else:
            # 多通道特征图，取平均值作为灰度表示
            gray = torch.mean(x, dim=-1)

        # 添加批次和通道维度
        gray = gray.reshape(B, 1, H, W)

        # 计算梯度
        grad_x = F.conv2d(gray, self.sobel_x.to(x.device), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(x.device), padding=1)

        # 计算梯度幅值
        edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # 调整形状
        edge_strength = edge_strength.reshape(B, H, W, 1)

        return edge_strength

    def forward(self, x, mask=None, edge_enhance=True):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            edge_enhance: 是否启用边缘增强
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 计算相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        # 添加边缘感知偏置（如果启用）
        if edge_enhance:
            # 将窗口特征转换回空间格式
            window_size = int(math.sqrt(N))
            x_spatial = x.view(-1, window_size, window_size, C)

            # 计算边缘特征
            edge_features = self.compute_edge_features(x_spatial)

            # 将边缘特征转换为注意力偏置
            edge_bias = self.edge_bias[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            edge_bias = edge_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]

            # 使用边缘特征加权边缘偏置
            edge_strength = edge_features.view(-1, window_size * window_size, 1)
            edge_strength = edge_strength.mean(dim=1, keepdim=True)  # 平均边缘强度 [B_, 1, 1]

            # 重新设计边缘偏置计算，确保维度匹配
            # 将edge_bias扩展到与edge_strength相同的批次维度
            expanded_edge_bias = edge_bias.unsqueeze(0)  # [1, num_heads, N, N]
            expanded_edge_strength = edge_strength.unsqueeze(1)  # [B_, 1, 1, 1]

            # 应用边缘强度加权
            enhanced_edge_bias = expanded_edge_bias * expanded_edge_strength  # [B_, num_heads, N, N]

            # 添加到注意力权重
            attn = attn + relative_position_bias.unsqueeze(0) + enhanced_edge_bias
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
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

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_edge_attention=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_edge_attention = use_edge_attention  # 新增参数
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        # self.attn = WindowAttention(
        #     dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # 修改为
        self.attn = WindowAttention(
            dim, num_heads=num_heads, window_size=self.window_size,  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


        # # 修改这里：使用EdgeAwareAttention替代WindowAttention
        # self.attn = EdgeAwareAttention(
        #     dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # # W-MSA/SW-MSA
        # # 修改这里：传递edge_enhance参数
        # if self.input_resolution == x_size:
        #     attn_windows = self.attn(x_windows, mask=self.attn_mask, edge_enhance=self.use_edge_attention)
        # else:
        #     attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device),
        #                              edge_enhance=self.use_edge_attention)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, use_edge_attention=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_edge_attention = use_edge_attention  # 新增参数

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_edge_attention=use_edge_attention)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
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


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv',use_edge_attention=False):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         use_edge_attention=use_edge_attention)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


# class SwinIR(nn.Module):
#     r""" SwinIR
#         A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
#
#     Args:
#         img_size (int | tuple(int)): Input image size. Default 64
#         patch_size (int | tuple(int)): Patch size. Default: 1
#         in_chans (int): Number of input image channels. Default: 3
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#         upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
#         img_range: Image range. 1. or 255.
#         upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
#         resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
#     """
#
#     def __init__(self, img_size=64, patch_size=1, in_chans=3,
#                  embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
#                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
#                  use_edge_attention=True,**kwargs):
#         super(SwinIR, self).__init__()
#         num_in_ch = in_chans
#         num_out_ch = in_chans
#         num_feat = 64
#         self.img_range = img_range
#         if in_chans == 3:
#             rgb_mean = (0.4488, 0.4371, 0.4040)
#             self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
#         else:
#             self.mean = torch.zeros(1, 1, 1, 1)
#         self.upscale = upscale
#         self.upsampler = upsampler
#         self.window_size = window_size
#
#         #####################################################################################################
#         ################################### 1, shallow feature extraction ###################################
#         self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
#
#         #####################################################################################################
#         ################################### 2, deep feature extraction ######################################
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = embed_dim
#         self.mlp_ratio = mlp_ratio
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         # merge non-overlapping patches into image
#         self.patch_unembed = PatchUnEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#
#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#
#         # build Residual Swin Transformer blocks (RSTB)
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = RSTB(dim=embed_dim,
#                          input_resolution=(patches_resolution[0],
#                                            patches_resolution[1]),
#                          depth=depths[i_layer],
#                          num_heads=num_heads[i_layer],
#                          window_size=window_size,
#                          mlp_ratio=self.mlp_ratio,
#                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                          drop=drop_rate, attn_drop=attn_drop_rate,
#                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
#                          norm_layer=norm_layer,
#                          downsample=None,
#                          use_checkpoint=use_checkpoint,
#                          img_size=img_size,
#                          patch_size=patch_size,
#                          resi_connection=resi_connection,
#                          use_edge_attention=use_edge_attention
#
#                          )
#             self.layers.append(layer)
#         self.norm = norm_layer(self.num_features)
#
#         # build the last conv layer in deep feature extraction
#         if resi_connection == '1conv':
#             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             # to save parameters and memory
#             self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
#
#         #####################################################################################################
#         ################################ 3, high quality image reconstruction ################################
#         if self.upsampler == 'pixelshuffle':
#             # for classical SR
#             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
#                                                       nn.LeakyReLU(inplace=True))
#             self.upsample = Upsample(upscale, num_feat)
#             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#         elif self.upsampler == 'pixelshuffledirect':
#             # for lightweight SR (to save parameters)
#             self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
#                                             (patches_resolution[0], patches_resolution[1]))
#         elif self.upsampler == 'nearest+conv':
#             # for real-world SR (less artifacts)
#             assert self.upscale == 4, 'only support x4 now.'
#             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
#                                                       nn.LeakyReLU(inplace=True))
#             self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         else:
#             # for image denoising and JPEG compression artifact reduction
#             self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
#         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#         return x
#
#     def forward_features(self, x):
#         x_size = (x.shape[2], x.shape[3])
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#
#         for layer in self.layers:
#             x = layer(x, x_size)
#
#         x = self.norm(x)  # B L C
#         x = self.patch_unembed(x, x_size)
#
#         return x
#
#     def forward(self, x):
#         H, W = x.shape[2:]
#         x = self.check_image_size(x)
#
#         self.mean = self.mean.type_as(x)
#         x = (x - self.mean) * self.img_range
#
#         if self.upsampler == 'pixelshuffle':
#             # for classical SR
#             x = self.conv_first(x)
#             x = self.conv_after_body(self.forward_features(x)) + x
#             x = self.conv_before_upsample(x)
#             x = self.conv_last(self.upsample(x))
#         elif self.upsampler == 'pixelshuffledirect':
#             # for lightweight SR
#             x = self.conv_first(x)
#             x = self.conv_after_body(self.forward_features(x)) + x
#             x = self.upsample(x)
#         elif self.upsampler == 'nearest+conv':
#             # for real-world SR
#             x = self.conv_first(x)
#             x = self.conv_after_body(self.forward_features(x)) + x
#             x = self.conv_before_upsample(x)
#             x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.conv_last(self.lrelu(self.conv_hr(x)))
#         else:
#             # for image denoising and JPEG compression artifact reduction
#             x_first = self.conv_first(x)
#             res = self.conv_after_body(self.forward_features(x_first)) + x_first
#             x = x + self.conv_last(res)
#
#         x = x / self.img_range + self.mean
#
#         return x[:, :, :H*self.upscale, :W*self.upscale]
#
#     def flops(self):
#         flops = 0
#         H, W = self.patches_resolution
#         flops += H * W * 3 * self.embed_dim * 9
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += H * W * 3 * self.embed_dim * self.embed_dim
#         flops += self.upsample.flops()
#         return flops



# class TripleInputSwinIR(nn.Module):
#     """支持三输入融合的SwinIR变体"""
#
#     def __init__(self, img_size=64, patch_size=1, in_chans=1,
#                  embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
#                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
#                  use_edge_attention=False, **kwargs):
#         super(TripleInputSwinIR, self).__init__()
#
#         # 基本参数
#         num_in_ch = in_chans
#         num_out_ch = in_chans
#         num_feat = 64
#         self.img_range = img_range
#         self.mean = torch.zeros(1, 1, 1, 1)  # 单通道数据的均值
#         self.upscale = upscale
#         self.upsampler = upsampler
#         self.window_size = window_size
#
#         #####################################################################################################
#         ################################### 1, 三输入融合层 ###################################
#         # 主输入处理
#         self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
#
#         # # 震源照明处理分支
#         # self.illumination_processor = nn.Sequential(
#         #     nn.Conv2d(1, 32, 3, 1, 1),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         #     nn.Conv2d(32, 32, 3, 1, 1),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         # )
#         #
#         # # 偏移速度模型处理分支
#         # self.velocity_processor = nn.Sequential(
#         #     nn.Conv2d(1, 32, 3, 1, 1),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         #     nn.Conv2d(32, 32, 3, 1, 1),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         # )
#         #
#         # # 特征融合层
#         # self.fusion_conv = nn.Conv2d(embed_dim + 32 + 32, embed_dim, 3, 1, 1)
#
#         # 震源照明处理分支 - 使用残差块
#         self.illumination_processor = nn.Sequential(
#             nn.Conv2d(1, 32, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             ResidualBlock(32),
#             ResidualBlock(32)
#         )
#
#         # 偏移速度模型处理分支 - 使用残差块
#         self.velocity_processor = nn.Sequential(
#             nn.Conv2d(1, 32, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             ResidualBlock(32),
#             ResidualBlock(32)
#         )
#
#         # 特征融合层 - 使用注意力机制
#         self.fusion_attention = nn.Sequential(
#             nn.Conv2d(embed_dim + 32 + 32, embed_dim, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.Sigmoid()
#         )
#         self.fusion_conv = nn.Conv2d(embed_dim + 32 + 32, embed_dim, 3, 1, 1)
#
#         #####################################################################################################
#         ################################### 2, 深層特征提取 ###################################
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = embed_dim
#         self.mlp_ratio = mlp_ratio
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         # merge non-overlapping patches into image
#         self.patch_unembed = PatchUnEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#
#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#
#         # build Residual Swin Transformer blocks (RSTB)
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = RSTB(dim=embed_dim,
#                          input_resolution=(patches_resolution[0],
#                                            patches_resolution[1]),
#                          depth=depths[i_layer],
#                          num_heads=num_heads[i_layer],
#                          window_size=window_size,
#                          mlp_ratio=self.mlp_ratio,
#                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                          drop=drop_rate, attn_drop=attn_drop_rate,
#                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                          norm_layer=norm_layer,
#                          downsample=None,
#                          use_checkpoint=use_checkpoint,
#                          img_size=img_size,
#                          patch_size=patch_size,
#                          resi_connection=resi_connection,
#                          use_edge_attention=use_edge_attention)
#             self.layers.append(layer)
#         self.norm = norm_layer(self.num_features)
#
#         # build the last conv layer in deep feature extraction
#         if resi_connection == '1conv':
#             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             # to save parameters and memory
#             self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
#
#         #####################################################################################################
#         ################################ 3, 高质量图像重建 ################################
#         if self.upsampler == 'pixelshuffle':
#             # for classical SR
#             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
#                                                       nn.LeakyReLU(inplace=True))
#             self.upsample = Upsample(upscale, num_feat)
#             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#         elif self.upsampler == 'pixelshuffledirect':
#             # for lightweight SR (to save parameters)
#             self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
#                                             (patches_resolution[0], patches_resolution[1]))
#         elif self.upsampler == 'nearest+conv':
#             # for real-world SR (less artifacts)
#             assert self.upscale == 4, 'only support x4 now.'
#             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
#                                                       nn.LeakyReLU(inplace=True))
#             self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         else:
#             # for image denoising and JPEG compression artifact reduction
#             self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
#         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#         return x
#
#     def forward_features(self, x):
#         x_size = (x.shape[2], x.shape[3])
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#
#         for layer in self.layers:
#             x = layer(x, x_size)
#
#         x = self.norm(x)  # B L C
#         x = self.patch_unembed(x, x_size)
#
#         return x
#
#     def forward(self, x, illumination=None, velocity=None):
#         # 三输入融合前向传播
#         H, W = x.shape[2:]
#         x = self.check_image_size(x)
#
#         self.mean = self.mean.type_as(x)
#         x = (x - self.mean) * self.img_range
#
#         # 处理震源照明输入
#         if illumination is not None:
#             illumination = self.check_image_size(illumination)
#             illumination_features = self.illumination_processor(illumination)
#         else:
#             illumination_features = torch.zeros_like(x)
#
#         # 处理偏移速度模型输入
#         if velocity is not None:
#             velocity = self.check_image_size(velocity)
#             velocity_features = self.velocity_processor(velocity)
#         else:
#             velocity_features = torch.zeros_like(x)
#
#         # 三输入融合
#         x = self.conv_first(x)
#         # 调整先验特征的尺寸（如果尺寸不匹配）
#         if illumination_features.shape[2:] != x.shape[2:]:
#             illumination_features = F.interpolate(illumination_features, size=x.shape[2:], mode='bilinear',
#                                                   align_corners=False)
#         if velocity_features.shape[2:] != x.shape[2:]:
#             velocity_features = F.interpolate(velocity_features, size=x.shape[2:], mode='bilinear',
#                                               align_corners=False)
#         combined_features = torch.cat([x, illumination_features, velocity_features], dim=1)
#         attention_weights = self.fusion_attention(combined_features)
#         # fused_features = self.fusion_conv(combined_features)
#         fused_features = attention_weights * self.fusion_conv(combined_features) + x  # 跳跃连接
#
#         # 深層特征提取
#         if self.upsampler == 'pixelshuffle':
#             # for classical SR
#             x = self.conv_after_body(self.forward_features(fused_features)) + fused_features
#             x = self.conv_before_upsample(x)
#             x = self.conv_last(self.upsample(x))
#         elif self.upsampler == 'pixelshuffledirect':
#             # for lightweight SR
#             x = self.conv_after_body(self.forward_features(fused_features)) + fused_features
#             x = self.upsample(x)
#         elif self.upsampler == 'nearest+conv':
#             # for real-world SR
#             x = self.conv_after_body(self.forward_features(fused_features)) + fused_features
#             x = self.conv_before_upsample(x)
#             x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.conv_last(self.lrelu(self.conv_hr(x)))
#         else:
#             # for image denoising and JPEG compression artifact reduction
#             x_first = fused_features
#             res = self.conv_after_body(self.forward_features(x_first)) + x_first
#             x = x_first + self.conv_last(res)
#
#         x = x / self.img_range + self.mean
#
#         return x[:, :, :H * self.upscale, :W * self.upscale]
#
#     def flops(self):
#         flops = 0
#         H, W = self.patches_resolution
#         flops += H * W * 3 * self.embed_dim * 9
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += H * W * 3 * self.embed_dim * self.embed_dim
#         flops += self.upsample.flops()
#         return flops
# # 残差块定义
# class ResidualBlock(nn.Module):
#     def __init__(self, n_feats):
#         super(ResidualBlock, self).__init__()
#         self.body = nn.Sequential(
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#         )
#         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return self.act(res)
# if __name__ == '__main__':
#     upscale = 4
#     window_size = 8
#     height = (1024 // upscale // window_size + 1) * window_size
#     width = (720 // upscale // window_size + 1) * window_size
#     model = TripleInputSwinIR(upscale=2, img_size=(height, width),
#                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
#                    embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
#     print(model)
#     print(height, width, model.flops() / 1e9)
#
#     x = torch.randn((1, 3, height, width))
#     x = model(x)
#     print(x.shape)

# superresolution-1-enhance-3-l1
# class PhysicsInformedFeatureFusion(nn.Module):
#     """基于地球物理原理的特征融合模块"""
#
#     def __init__(self, embed_dim):
#         super(PhysicsInformedFeatureFusion, self).__init__()
#         self.embed_dim = embed_dim
#
#         # 照明增强模块 - 模拟地震照明效应
#         self.illumination_enhancer = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.Sigmoid()  # 输出在0-1之间，作为权重
#         )
#
#         # 速度约束模块 - 模拟速度对波传播的影响
#         self.velocity_constraint = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#         )
#
#         # 多尺度特征提取 - 捕捉不同尺度的地质特征
#         self.multi_scale_extract = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, 1, 1)
#             ) for _ in range(3)
#         ])
#
#         # 多尺度特征融合
#         self.scale_fusion = nn.Sequential(
#             nn.Conv2d(embed_dim + embed_dim // 4 * 3, embed_dim, 1, 1, 0),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         )
#
#         # 物理约束注意力机制
#         self.physics_attention = nn.Sequential(
#             nn.Conv2d(embed_dim * 3, embed_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.Sigmoid()
#         )
#
#         # 最终融合
#         self.final_fusion = nn.Sequential(
#             nn.Conv2d(embed_dim * 3, embed_dim * 2, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)
#         )
#
#     def forward(self, x_main, illum_features, velocity_features):
#         # 应用照明增强 - 模拟地震照明效应
#         illumination_weights = self.illumination_enhancer(illum_features)
#         x_illuminated = x_main * illumination_weights
#
#         # 应用速度约束 - 模拟速度场对波传播的影响
#         velocity_constraint = self.velocity_constraint(velocity_features)
#         x_velocity_constrained = x_illuminated + velocity_constraint
#
#         # 多尺度特征提取 - 捕捉不同尺度的地质特征
#         scale_features = []
#         for i, extractor in enumerate(self.multi_scale_extract):
#             scale_factor = 2 ** (i + 1)
#             scaled = F.interpolate(x_velocity_constrained, scale_factor=1 / scale_factor,
#                                    mode='bilinear', align_corners=False)
#             feat = extractor(scaled)
#             feat = F.interpolate(feat, size=x_velocity_constrained.shape[2:],
#                                  mode='bilinear', align_corners=False)
#             scale_features.append(feat)
#
#         # 融合多尺度特征
#         multi_scale_feat = torch.cat([x_velocity_constrained] + scale_features, dim=1)
#         x_multi_scale = self.scale_fusion(multi_scale_feat)
#
#         # 物理约束注意力机制
#         attention_input = torch.cat([x_multi_scale, illum_features, velocity_features], dim=1)
#         attention_weights = self.physics_attention(attention_input)
#
#         # 应用注意力权重
#         attended_features = x_multi_scale * attention_weights
#
#         # 最终融合
#         final_combined = torch.cat([attended_features, illum_features, velocity_features], dim=1)
#         output = self.final_fusion(final_combined) + x_main  # 残差连接
#
#         return output
#
#
# class EnhancedTripleInputSwinIR(nn.Module):
#     """改进的三输入融合SwinIR变体 - 结合乘法融合和注意力重加权"""
#
#     def __init__(self, img_size=128, patch_size=1, in_chans=1,
#                  embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
#                  window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
#                  use_edge_attention=False, **kwargs):
#         super(EnhancedTripleInputSwinIR, self).__init__()
#
#         # 基本参数
#         num_in_ch = in_chans
#         num_out_ch = in_chans
#         num_feat = 64
#         self.img_range = img_range
#         self.mean = torch.zeros(1, 1, 1, 1)
#         self.upscale = upscale
#         self.upsampler = upsampler
#         self.window_size = window_size
#
#         #####################################################################################################
#         ################################### 1, 三输入融合层 ###################################
#         # 主输入处理
#         self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
#
#         # 震源照明处理分支 - 使用更深的网络
#         self.illumination_processor = nn.Sequential(
#             nn.Conv2d(1, 128, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(64, embed_dim, 3, 1, 1)  # 直接输出与主特征相同的维度
#         )
#
#         # 偏移速度模型处理分支 - 使用更深的网络
#         self.velocity_processor = nn.Sequential(
#             nn.Conv2d(1, 128, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(64, embed_dim, 3, 1, 1)  # 直接输出与主特征相同的维度
#         )
#
#
#         # 使用增强的特征融合模块
#         self.feature_fusion = PhysicsInformedFeatureFusion(embed_dim)
#
#         #####################################################################################################
#         ################################### 2, 深層特征提取 ###################################
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = embed_dim
#         self.mlp_ratio = mlp_ratio
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         # merge non-overlapping patches into image
#         self.patch_unembed = PatchUnEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#
#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#
#         # build Residual Swin Transformer blocks (RSTB)
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = RSTB(dim=embed_dim,
#                          input_resolution=(patches_resolution[0],
#                                            patches_resolution[1]),
#                          depth=depths[i_layer],
#                          num_heads=num_heads[i_layer],
#                          window_size=window_size,
#                          mlp_ratio=self.mlp_ratio,
#                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                          drop=drop_rate, attn_drop=attn_drop_rate,
#                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                          norm_layer=norm_layer,
#                          downsample=None,
#                          use_checkpoint=use_checkpoint,
#                          img_size=img_size,
#                          patch_size=patch_size,
#                          resi_connection=resi_connection,
#                          use_edge_attention=use_edge_attention)
#             self.layers.append(layer)
#         self.norm = norm_layer(self.num_features)
#
#         # build the last conv layer in deep feature extraction
#         if resi_connection == '1conv':
#             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
#
#         #####################################################################################################
#         ################################ 3, 高质量图像重建 ################################
#         if self.upsampler == 'pixelshuffle':
#             # self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
#             #                                           nn.LeakyReLU(inplace=True))
#             # self.upsample = Upsample(upscale, num_feat)
#             # self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             # 增强的上采样路径
#             self.conv_before_upsample = nn.Sequential(
#                 nn.Conv2d(embed_dim, num_feat * 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True)
#             )
#             self.upsample = Upsample(upscale, num_feat * 4)
#             self.conv_last = nn.Sequential(
#                 nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             )
#
#         elif self.upsampler == 'pixelshuffledirect':
#             self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
#                                             (patches_resolution[0], patches_resolution[1]))
#         elif self.upsampler == 'nearest+conv':
#             assert self.upscale == 4, 'only support x4 now.'
#             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
#                                                       nn.LeakyReLU(inplace=True))
#             self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         else:
#             self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
#         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#         return x
#
#     def forward_features(self, x):
#         x_size = (x.shape[2], x.shape[3])
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#
#         for layer in self.layers:
#             x = layer(x, x_size)
#
#         x = self.norm(x)
#         x = self.patch_unembed(x, x_size)
#
#         return x
#
#     def forward(self, x, illumination=None, velocity=None):
#         # 三输入融合前向传播
#         H, W = x.shape[2:]
#         x = self.check_image_size(x)
#
#         self.mean = self.mean.type_as(x)
#         x = (x - self.mean) * self.img_range
#
#         # 处理主输入
#         x_main = self.conv_first(x)
#
#         # 处理震源照明输入 (方案一: 乘法融合)
#         if illumination is not None:
#             illumination = self.check_image_size(illumination)
#             # # 归一化以提高数值稳定性
#             # illumination = (illumination - torch.mean(illumination)) / (torch.std(illumination) + 1e-8)
#             illumination_features = self.illumination_processor(illumination)
#             # # 调整通道数
#             # illumination_features_adjusted = self.illumination_channel_adjust(illumination_features)
#             # 调整尺寸
#             if illumination_features.shape[2:] != x_main.shape[2:]:
#                 illumination_features_adjusted = F.interpolate(illumination_features,
#                                                                size=x_main.shape[2:],
#                                                                mode='bilinear', align_corners=False)
#             # # 乘法融合: 主特征 × 照明特征
#             # x_illuminated = x_main * illumination_features_adjusted
#         else:
#             illumination_features = torch.ones_like(x_main)  # 使用1而不是0
#             # x_illuminated = x_main
#
#
#         # 处理偏移速度模型输入
#         if velocity is not None:
#             velocity = self.check_image_size(velocity)
#             # velocity = (velocity - torch.mean(velocity)) / (torch.std(velocity) + 1e-8)
#             velocity_features = self.velocity_processor(velocity)
#             # # 调整通道数
#             # velocity_features_adjusted = self.velocity_channel_adjust(velocity_features)
#             # 调整尺寸
#             if velocity_features.shape[2:] != x_main.shape[2:]:
#                 velocity_features_adjusted = F.interpolate(velocity_features,
#                                                            size=x_main.shape[2:],
#                                                            mode='bilinear', align_corners=False)
#         else:
#             velocity_features = torch.zeros_like(x_main)
#
#         # # 使用增强的特征融合模块 (符合GAN做法: 相乘结果与偏移速度模型拼接)
#         final_fused = self.feature_fusion(x_main, illumination_features, velocity_features)
#
#         # 深層特征提取
#         if self.upsampler == 'pixelshuffle':
#             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
#             x = self.conv_before_upsample(x)
#             x = self.conv_last(self.upsample(x))
#         elif self.upsampler == 'pixelshuffledirect':
#             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
#             x = self.upsample(x)
#         elif self.upsampler == 'nearest+conv':
#             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
#             x = self.conv_before_upsample(x)
#             x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.conv_last(self.lrelu(self.conv_hr(x)))
#         else:
#             x_first = final_fused
#             res = self.conv_after_body(self.forward_features(x_first)) + x_first
#             x = x_first + self.conv_last(res)
#
#         x = x / self.img_range + self.mean
#
#         return x[:, :, :H * self.upscale, :W * self.upscale]
#
#     def flops(self):
#         flops = 0
#         H, W = self.patches_resolution
#         flops += H * W * 3 * self.embed_dim * 9
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += H * W * 3 * self.embed_dim * self.embed_dim
#         flops += self.upsample.flops()
#         return flops
#
#
# # # 残差块定义
# # class ResidualBlock(nn.Module):
# #     def __init__(self, n_feats):
# #         super(ResidualBlock, self).__init__()
# #         self.body = nn.Sequential(
# #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# #         )
# #         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# #
# #     def forward(self, x):
# #         res = self.body(x)
# #         res += x
# #         return self.act(res)
#
# # 增强的残差块
# class ResidualBlock(nn.Module):
#     def __init__(self, n_feats, reduction=16):
#         super(ResidualBlock, self).__init__()
#         self.body = nn.Sequential(
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#         )
#
#         # 通道注意力
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(n_feats, n_feats // reduction, 1, 1, 0),
#             nn.ReLU(True),
#             nn.Conv2d(n_feats // reduction, n_feats, 1, 1, 0),
#             nn.Sigmoid()
#         )
#
#         # 空间注意力
#         self.sa = nn.Sequential(
#             nn.Conv2d(2, 1, 7, 1, 3),
#             nn.Sigmoid()
#         )
#
#         self.act = nn.LeakyReLU(0.2, True)
#
#     def forward(self, x):
#         res = self.body(x)
#
#         # 通道注意力
#         ca_weights = self.ca(res)
#         res = res * ca_weights
#
#         # 空间注意力
#         avg_out = torch.mean(res, dim=1, keepdim=True)
#         max_out, _ = torch.max(res, dim=1, keepdim=True)
#         sa_weights = self.sa(torch.cat([avg_out, max_out], dim=1))
#         res = res * sa_weights
#
#         res += x
#         return self.act(res)
#
# if __name__ == '__main__':
#     upscale = 4
#     window_size = 8
#     height = (1024 // upscale // window_size + 1) * window_size
#     width = (720 // upscale // window_size + 1) * window_size
#     model = EnhancedTripleInputSwinIR(upscale=2, img_size=(height, width),
#                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
#                    embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
#     print(model)
#     print(height, width, model.flops() / 1e9)
#
#     x = torch.randn((1, 3, height, width))
#     x = model(x)
#     print(x.shape)




#3inputs---------important
class PhysicsInformedFeatureFusion(nn.Module):
   """基于地球物理原理的特征融合模块"""

   def __init__(self, embed_dim):
       super(PhysicsInformedFeatureFusion, self).__init__()
       self.embed_dim = embed_dim

       # 照明增强模块 - 模拟地震照明效应
       self.illumination_enhancer = nn.Sequential(
           nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
           nn.Sigmoid()  # 输出在0-1之间，作为权重
       )

       # 速度约束模块 - 模拟速度对波传播的影响
       self.velocity_constraint = nn.Sequential(
           nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
       )


       # 多尺度特征提取 - 捕捉不同尺度的地质特征
       self.multi_scale_extract = nn.ModuleList([
           nn.Sequential(
               nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
               nn.LeakyReLU(0.2, True),
               nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, 1, 1)
           ) for _ in range(3)
       ])

       # 多尺度特征融合
       self.scale_fusion = nn.Sequential(
           nn.Conv2d(embed_dim + embed_dim // 4 * 3, embed_dim, 1, 1, 0),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
       )

       # 物理约束注意力机制
       self.physics_attention = nn.Sequential(
           nn.Conv2d(embed_dim * 3, embed_dim, 3, 1, 1),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
           nn.Sigmoid()
       )

       # 最终融合
       self.final_fusion = nn.Sequential(
           nn.Conv2d(embed_dim * 3, embed_dim * 2, 3, 1, 1),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)
       )



   def forward(self, x_main, illum_features, velocity_features):
       # print(f"DEBUG - x_main shape: {x_main.shape}")
       # print(f"DEBUG - illum_features shape: {illum_features.shape}")
       # print(f"DEBUG - velocity_features shape: {velocity_features.shape}")

       # 保存原始输入用于残差连接
       identity = x_main
       # # 关键修复：确保所有输入尺寸一致
       # if illum_features.shape[2:] != x_main.shape[2:]:
       #     illum_features = F.interpolate(illum_features,
       #                                    size=x_main.shape[2:],
       #                                    mode='bilinear', align_corners=False)
       #
       # if velocity_features.shape[2:] != x_main.shape[2:]:
       #     velocity_features = F.interpolate(velocity_features,
       #                                       size=x_main.shape[2:],
       #                                       mode='bilinear', align_corners=False)
       # # 由于在主模型中已经确保尺寸一致，这里可以添加断言而不是重复调整
       # assert illum_features.shape[2:] == x_main.shape[2:], \
       #     f"Illumination features size mismatch: {illum_features.shape[2:]} vs {x_main.shape[2:]}"
       # assert velocity_features.shape[2:] == x_main.shape[2:], \
       #     f"Velocity features size mismatch: {velocity_features.shape[2:]} vs {x_main.shape[2:]}"

       # 应用照明增强 - 模拟地震照明效应
       illumination_weights = self.illumination_enhancer(illum_features)
       x_illuminated = x_main * illumination_weights

       # 应用速度约束 - 模拟速度场对波传播的影响
       velocity_constraint = self.velocity_constraint(velocity_features)
       x_velocity_constrained = x_illuminated + velocity_constraint


       # 多尺度特征提取 - 捕捉不同尺度的地质特征
       scale_features = []
       for i, extractor in enumerate(self.multi_scale_extract):
           scale_factor = 2 ** (i + 1)
           scaled = F.interpolate(x_velocity_constrained, scale_factor=1 / scale_factor,
                                  mode='bilinear', align_corners=False)
           feat = extractor(scaled)
           feat = F.interpolate(feat, size=x_velocity_constrained.shape[2:],
                                mode='bilinear', align_corners=False)
           scale_features.append(feat)

       # 融合多尺度特征
       multi_scale_feat = torch.cat([x_velocity_constrained] + scale_features, dim=1)
       x_multi_scale = self.scale_fusion(multi_scale_feat)

       # 物理约束注意力机制
       attention_input = torch.cat([x_multi_scale, illum_features, velocity_features], dim=1)
       attention_weights = self.physics_attention(attention_input)

       # 应用注意力权重
       attended_features = x_multi_scale * attention_weights

       # 最终融合
       final_combined = torch.cat([attended_features, illum_features, velocity_features], dim=1)
       output = self.final_fusion(final_combined)
       # 强有力的残差连接：最终输出必须包含原始的 x_main
       output = output + identity

       return output


class EnhancedTripleInputSwinIR(nn.Module):
   """改进的三输入融合SwinIR变体 - 结合乘法融合和注意力重加权"""

   def set_alpha(self, alpha):
       """设置先验特征的权重系数"""
       self.alpha = alpha

   def __init__(self, img_size=128, patch_size=1, in_chans=1,
                embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                use_edge_attention=False, **kwargs):
       super(EnhancedTripleInputSwinIR, self).__init__()

       # 基本参数
       num_in_ch = in_chans
       num_out_ch = in_chans
       num_feat = 64
       self.img_range = img_range
       self.mean = torch.zeros(1, 1, 1, 1)
       self.upscale = upscale
       self.upsampler = upsampler
       self.window_size = window_size
       self.alpha = 1.0  # 默认权重为1.0
       #####################################################################################################
       ################################### 1, 三输入融合层 ###################################
       # 主输入处理
       self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

       # 震源照明处理分支 - 使用更深的网络
       self.illumination_processor = nn.Sequential(
           nn.Conv2d(1, 128, 3, 1, 1),
           nn.LeakyReLU(negative_slope=0.2, inplace=True),
           ResidualBlock(128),
           ResidualBlock(128),
           ResidualBlock(128),
           nn.Conv2d(128, 64, 3, 1, 1),
           nn.LeakyReLU(negative_slope=0.2, inplace=True),
           nn.Conv2d(64, embed_dim, 3, 1, 1)  # 直接输出与主特征相同的维度
       )

       # 偏移速度模型处理分支 - 使用更深的网络
       self.velocity_processor = nn.Sequential(
           nn.Conv2d(1, 128, 3, 1, 1),
           nn.LeakyReLU(negative_slope=0.2, inplace=True),
           ResidualBlock(128),
           ResidualBlock(128),
           ResidualBlock(128),
           nn.Conv2d(128, 64, 3, 1, 1),
           nn.LeakyReLU(negative_slope=0.2, inplace=True),
           nn.Conv2d(64, embed_dim, 3, 1, 1)  # 直接输出与主特征相同的维度
       )

       # # 简化的先验信息处理器
       # self.illumination_processor = nn.Sequential(
       #     nn.Conv2d(1, 64, 3, 1, 1),
       #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
       #     nn.Conv2d(64, embed_dim, 3, 1, 1)
       # )
       #
       # self.velocity_processor = nn.Sequential(
       #     nn.Conv2d(1, 64, 3, 1, 1),
       #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
       #     nn.Conv2d(64, embed_dim, 3, 1, 1)
       # )

       # 使用增强的特征融合模块
       self.feature_fusion = PhysicsInformedFeatureFusion(embed_dim)

       #####################################################################################################
       ################################### 2, 深層特征提取 ###################################
       self.num_layers = len(depths)
       self.embed_dim = embed_dim
       self.ape = ape
       self.patch_norm = patch_norm
       self.num_features = embed_dim
       self.mlp_ratio = mlp_ratio

       # split image into non-overlapping patches
       self.patch_embed = PatchEmbed(
           img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
           norm_layer=norm_layer if self.patch_norm else None)
       num_patches = self.patch_embed.num_patches
       patches_resolution = self.patch_embed.patches_resolution
       self.patches_resolution = patches_resolution

       # merge non-overlapping patches into image
       self.patch_unembed = PatchUnEmbed(
           img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
           norm_layer=norm_layer if self.patch_norm else None)

       # absolute position embedding
       if self.ape:
           self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
           trunc_normal_(self.absolute_pos_embed, std=.02)

       self.pos_drop = nn.Dropout(p=drop_rate)

       # stochastic depth
       dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

       # build Residual Swin Transformer blocks (RSTB)
       self.layers = nn.ModuleList()
       for i_layer in range(self.num_layers):
           layer = RSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0],
                                          patches_resolution[1]),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint,
                        img_size=img_size,
                        patch_size=patch_size,
                        resi_connection=resi_connection,
                        use_edge_attention=use_edge_attention)
           self.layers.append(layer)
       self.norm = norm_layer(self.num_features)

       # build the last conv layer in deep feature extraction
       if resi_connection == '1conv':
           self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
       elif resi_connection == '3conv':
           self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

       #####################################################################################################
       ################################ 3, 高质量图像重建 ################################
       if self.upsampler == 'pixelshuffle':
           self.conv_before_upsample = nn.Sequential(
               nn.Conv2d(embed_dim, num_feat * 4, 3, 1, 1),
               nn.LeakyReLU(0.2, True),
               nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
               nn.LeakyReLU(0.2, True),
               nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
               nn.LeakyReLU(0.2, True)
           )
           self.upsample = Upsample(upscale, num_feat * 4)
           self.conv_last = nn.Sequential(
               nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1),
               nn.LeakyReLU(0.2, True),
               nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
               nn.LeakyReLU(0.2, True),
               nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
       )

           # self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
           #                                           nn.LeakyReLU(inplace=True))
           # self.upsample = Upsample(upscale, num_feat)
           # self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
           # # 增强的上采样路径
           # self.conv_before_upsample = nn.Sequential(
           #     nn.Conv2d(embed_dim, num_feat , 3, 1, 1),
           #     nn.LeakyReLU(0.2, True),
           #     nn.Conv2d(num_feat , num_feat , 3, 1, 1),
           #     nn.LeakyReLU(0.2, True),
           #     nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
           #     nn.LeakyReLU(0.2, True)
           # )
           # self.upsample = Upsample(upscale, num_feat )
           # self.conv_last = nn.Sequential(
           #     nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1),
           #     nn.LeakyReLU(0.2, True),
           #     nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
           #     nn.LeakyReLU(0.2, True),
           #     nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
           # )


       elif self.upsampler == 'pixelshuffledirect':
           self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                           (patches_resolution[0], patches_resolution[1]))
       elif self.upsampler == 'nearest+conv':
           assert self.upscale == 4, 'only support x4 now.'
           self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                     nn.LeakyReLU(inplace=True))
           self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
           self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
           self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
           self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
           self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
       else:
           self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

       self.apply(self._init_weights)

   def _init_weights(self, m):
       if isinstance(m, nn.Linear):
           trunc_normal_(m.weight, std=.02)
           if isinstance(m, nn.Linear) and m.bias is not None:
               nn.init.constant_(m.bias, 0)
       elif isinstance(m, nn.LayerNorm):
           nn.init.constant_(m.bias, 0)
           nn.init.constant_(m.weight, 1.0)


   @torch.jit.ignore
   def no_weight_decay(self):
       return {'absolute_pos_embed'}

   @torch.jit.ignore
   def no_weight_decay_keywords(self):
       return {'relative_position_bias_table'}

   def check_image_size(self, x):
       _, _, h, w = x.size()
       mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
       mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
       x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
       return x

   def forward_features(self, x):
       x_size = (x.shape[2], x.shape[3])
       x = self.patch_embed(x)
       if self.ape:
           x = x + self.absolute_pos_embed
       x = self.pos_drop(x)

       for layer in self.layers:
           x = layer(x, x_size)

       x = self.norm(x)
       x = self.patch_unembed(x, x_size)

       return x

   def forward(self, x, illumination=None, velocity=None):
       # 三输入融合前向传播
       H, W = x.shape[2:]
       x = self.check_image_size(x)

       self.mean = self.mean.type_as(x)
       x = (x - self.mean) * self.img_range

       # 处理主输入
       x_main = self.conv_first(x)

       # 处理震源照明输入 (方案一: 乘法融合)
       if illumination is not None:
           illumination = self.check_image_size(illumination)
           # # 归一化以提高数值稳定性
           # illumination = (illumination - torch.mean(illumination)) / (torch.std(illumination) + 1e-8)
           illumination_features = self.illumination_processor(illumination)
           # # 调整通道数
           # illumination_features_adjusted = self.illumination_channel_adjust(illumination_features)
           # # 应用渐进式训练的alpha系数
           # illumination_features = illumination_features * self.alpha
           # 调整尺寸
           if illumination_features.shape[2:] != x_main.shape[2:]:
               illumination_features = F.interpolate(illumination_features,
                                                              size=x_main.shape[2:],
                                                              mode='bilinear', align_corners=False)
           # # 乘法融合: 主特征 × 照明特征
           # x_illuminated = x_main * illumination_features_adjusted
       else:
           illumination_features = torch.ones_like(x_main)*self.alpha  # 使用1而不是0
           # x_illuminated = x_main


       # 处理偏移速度模型输入
       if velocity is not None:
           velocity = self.check_image_size(velocity)
           # velocity = (velocity - torch.mean(velocity)) / (torch.std(velocity) + 1e-8)
           velocity_features = self.velocity_processor(velocity)
           # # 调整通道数
           # velocity_features_adjusted = self.velocity_channel_adjust(velocity_features)
           # # 应用渐进式训练的alpha系数
           # velocity_features = velocity_features * self.alpha
           # 调整尺寸
           if velocity_features.shape[2:] != x_main.shape[2:]:
               velocity_features = F.interpolate(velocity_features,
                                                          size=x_main.shape[2:],
                                                          mode='bilinear', align_corners=False)
       else:
           velocity_features = torch.zeros_like(x_main)

       # # 使用增强的特征融合模块 (符合GAN做法: 相乘结果与偏移速度模型拼接)
       final_fused = self.feature_fusion(x_main, illumination_features, velocity_features)

       # 深層特征提取
       if self.upsampler == 'pixelshuffle':

           x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
           x = self.conv_before_upsample(x)
           x = self.conv_last(self.upsample(x))
       elif self.upsampler == 'pixelshuffledirect':
           x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
           x = self.upsample(x)
       elif self.upsampler == 'nearest+conv':
           x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
           x = self.conv_before_upsample(x)
           x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
           x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
           x = self.conv_last(self.lrelu(self.conv_hr(x)))
       else:
           x_first = final_fused
           res = self.conv_after_body(self.forward_features(x_first)) + x_first
           x = x_first + self.conv_last(res)

       x = x / self.img_range + self.mean

       return x[:, :, :H * self.upscale, :W * self.upscale]

   def flops(self):
       flops = 0
       H, W = self.patches_resolution
       flops += H * W * 3 * self.embed_dim * 9
       flops += self.patch_embed.flops()
       for i, layer in enumerate(self.layers):
           flops += layer.flops()
       flops += H * W * 3 * self.embed_dim * self.embed_dim
       flops += self.upsample.flops()
       return flops


# # 残差块定义
# class ResidualBlock(nn.Module):
#     def __init__(self, n_feats):
#         super(ResidualBlock, self).__init__()
#         self.body = nn.Sequential(
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#         )
#         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return self.act(res)

# 增强的残差块
class ResidualBlock(nn.Module):
   def __init__(self, n_feats, reduction=16):
       super(ResidualBlock, self).__init__()
       self.body = nn.Sequential(
           nn.Conv2d(n_feats, n_feats, 3, 1, 1),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(n_feats, n_feats, 3, 1, 1),
       )

       # 通道注意力
       self.ca = nn.Sequential(
           nn.AdaptiveAvgPool2d(1),
           nn.Conv2d(n_feats, n_feats // reduction, 1, 1, 0),
           nn.ReLU(True),
           nn.Conv2d(n_feats // reduction, n_feats, 1, 1, 0),
           nn.Sigmoid()
       )

       # 空间注意力
       self.sa = nn.Sequential(
           nn.Conv2d(2, 1, 7, 1, 3),
           nn.Sigmoid()
       )

       self.act = nn.LeakyReLU(0.2, True)

   def forward(self, x):
       res = self.body(x)

       # 通道注意力
       ca_weights = self.ca(res)
       res = res * ca_weights

       # 空间注意力
       avg_out = torch.mean(res, dim=1, keepdim=True)
       max_out, _ = torch.max(res, dim=1, keepdim=True)
       sa_weights = self.sa(torch.cat([avg_out, max_out], dim=1))
       res = res * sa_weights

       res += x
       return self.act(res)

if __name__ == '__main__':
   upscale = 4
   window_size = 8
   height = (1024 // upscale // window_size + 1) * window_size
   width = (720 // upscale // window_size + 1) * window_size
   model = EnhancedTripleInputSwinIR(upscale=2, img_size=(height, width),
                  window_size=window_size, img_range=1., depths=[6, 6, 6, 6,6,6],
                  embed_dim=180, num_heads=[6, 6, 6, 6,6,6], mlp_ratio=2, upsampler='pixelshuffledirect')
   print(model)
   print(height, width, model.flops() / 1e9)

   x = torch.randn((1, 3, height, width))
   x = model(x)
   print(x.shape)


# # 哪个为0就不用哪个
# # 增强的残差块
# class ResidualBlock(nn.Module):
#     def __init__(self, n_feats, reduction=16):
#         super(ResidualBlock, self).__init__()
#         self.body = nn.Sequential(
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#         )
#
#         # 通道注意力
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(n_feats, n_feats // reduction, 1, 1, 0),
#             nn.ReLU(True),
#             nn.Conv2d(n_feats // reduction, n_feats, 1, 1, 0),
#             nn.Sigmoid()
#         )
#
#         # 空间注意力
#         self.sa = nn.Sequential(
#             nn.Conv2d(2, 1, 7, 1, 3),
#             nn.Sigmoid()
#         )
#
#         self.act = nn.LeakyReLU(0.2, True)
#
#     def forward(self, x):
#         res = self.body(x)
#
#         # 通道注意力
#         ca_weights = self.ca(res)
#         res = res * ca_weights
#
#         # 空间注意力
#         avg_out = torch.mean(res, dim=1, keepdim=True)
#         max_out, _ = torch.max(res, dim=1, keepdim=True)
#         sa_weights = self.sa(torch.cat([avg_out, max_out], dim=1))
#         res = res * sa_weights
#
#         res += x
#         return self.act(res)
#
#
# class PhysicsInformedFeatureFusion(nn.Module):
#     """基于地球物理原理的特征融合模块
#     已修改：自适应支持照明/速度任一为0时自动禁用对应分支
#     """
#
#     def __init__(self, embed_dim):
#         super(PhysicsInformedFeatureFusion, self).__init__()
#         self.embed_dim = embed_dim
#
#         # 照明增强模块 - 模拟地震照明效应
#         self.illumination_enhancer = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.Sigmoid()  # 输出在0-1之间，作为权重
#         )
#
#         # 速度约束模块 - 模拟速度对波传播的影响
#         self.velocity_constraint = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#         )
#
#         # 多尺度特征提取 - 捕捉不同尺度的地质特征
#         self.multi_scale_extract = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, 1, 1)
#             ) for _ in range(3)
#         ])
#
#         # 多尺度特征融合
#         self.scale_fusion = nn.Sequential(
#             nn.Conv2d(embed_dim + embed_dim // 4 * 3, embed_dim, 1, 1, 0),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         )
#
#         # 【修改1】注意力适配最大2路输入
#         self.physics_attention = nn.Sequential(
#             nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.Sigmoid()
#         )
#
#         # 【修改2】最终融合适配2路通道
#         self.final_fusion = nn.Sequential(
#             nn.Conv2d(embed_dim * 2, embed_dim * 2, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)
#         )
#
#     # 【修改3】前向传播自适应判断是否存在有效先验
#     def forward(self, x_main, illum_features, velocity_features):
#         # 保存原始输入用于残差连接
#         identity = x_main
#
#         # 判断是否为有效非零先验
#         has_illum = not torch.all(torch.abs(illum_features) < 1e-8)
#         has_vel = not torch.all(torch.abs(velocity_features) < 1e-8)
#
#         # 多尺度分支只在有至少一个先验时走
#         if has_illum or has_vel:
#             # 多尺度特征提取
#             scale_features = []
#             for i, extractor in enumerate(self.multi_scale_extract):
#                 scale_factor = 2 ** (i + 1)
#                 scaled = F.interpolate(x_main, scale_factor=1 / scale_factor,
#                                        mode='bilinear', align_corners=False)
#                 feat = extractor(scaled)
#                 feat = F.interpolate(feat, size=x_main.shape[2:],
#                                      mode='bilinear', align_corners=False)
#                 scale_features.append(feat)
#             multi_scale_feat = torch.cat([x_main] + scale_features, dim=1)
#             x_multi_scale = self.scale_fusion(multi_scale_feat)
#         else:
#             x_multi_scale = x_main
#
#         # 情况1：同时有照明和速度先验
#         if has_illum and has_vel:
#             illum_w = self.illumination_enhancer(illum_features)
#             x_illu = x_multi_scale * illum_w
#             vel_const = self.velocity_constraint(velocity_features)
#             x_comb = x_illu + vel_const
#
#             att_inp = torch.cat([x_comb, illum_features], dim=1)
#             att_w = self.physics_attention(att_inp)
#             attended = x_comb * att_w
#
#             final_comb = torch.cat([attended, velocity_features], dim=1)
#             output = self.final_fusion(final_comb)
#
#         # 情况2：只有照明先验，无速度
#         elif has_illum:
#             illum_w = self.illumination_enhancer(illum_features)
#             x_comb = x_multi_scale + x_multi_scale * illum_w
#
#             att_inp = torch.cat([x_comb, illum_features], dim=1)
#             att_w = self.physics_attention(att_inp)
#             attended = x_comb * att_w
#             final_comb = torch.cat([attended, illum_features], dim=1)
#             output = self.final_fusion(final_comb)
#
#         # 情况3：只有速度先验，无照明
#         elif has_vel:
#             vel_const = self.velocity_constraint(velocity_features)
#             x_comb = x_multi_scale + vel_const
#
#             att_inp = torch.cat([x_comb, velocity_features], dim=1)
#             att_w = self.physics_attention(att_inp)
#             attended = x_comb * att_w
#             final_comb = torch.cat([attended, velocity_features], dim=1)
#             output = self.final_fusion(final_comb)
#
#         # 情况4：两个先验都为0，只用主特征
#         else:
#             output = x_multi_scale
#
#         # 残差连接
#         output = output + identity
#         return output
#
#
# # ===================== SwinIR 基础模块 =====================
# class PatchEmbed(nn.Module):
#     def __init__(self, img_size=128, patch_size=1, in_chans=1, embed_dim=180, norm_layer=None):
#         super().__init__()
#         img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
#         self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else None
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         if self.norm is not None:
#             x = self.norm(x)
#         return x
#
#
# class PatchUnEmbed(nn.Module):
#     def __init__(self, img_size=128, patch_size=1, in_chans=1, embed_dim=180, norm_layer=None):
#         super().__init__()
#         img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
#
#     def forward(self, x, x_size):
#         B, HW, C = x.shape
#         x = x.transpose(1, 2).view(B, C, x_size[0], x_size[1])
#         return x
#
#
# class WindowAttention(nn.Module):
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
#         coords_flatten = torch.flatten(coords, 1)
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#         relative_coords[:, :, 0] += self.window_size[0] - 1
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)
#         self.register_buffer("relative_position_index", relative_position_index)
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#
#     def forward(self, x, mask=None):
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         attn = attn + relative_position_bias.unsqueeze(0)
#
#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# def window_partition(x, window_size):
#     B, C, H, W = x.shape
#     x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
#     windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
#     windows = windows.view(-1, window_size * window_size, C)
#     return windows
#
#
# def window_reverse(windows, window_size, H, W):
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
#     x = x.view(B, -1, H, W)
#     return x
#
#
# class SwinTransformerBlock(nn.Module):
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size=8,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
#                  use_edge_attention=False):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.mlp_ratio = mlp_ratio
#         self.use_checkpoint = use_checkpoint
#
#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#
#         self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(drop)
#         )
#
#     # def forward(self, x):
#     #     H, W = self.input_resolution
#     #     B, L, C = x.shape
#     #     assert L == H * W, "input feature has wrong size"
#
#     def forward(self, x):
#         B, L, C = x.shape
#         H = W = int(L ** 0.5)  # 假设特征图为正方形
#         assert H * W == L, f"input feature has wrong size: L={L}, H*W={H * W}"
#
#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C).permute(0, 3, 1, 2)
#
#         pad_l = pad_t = 0
#         pad_r = (self.window_size - W % self.window_size) % self.window_size
#         pad_b = (self.window_size - H % self.window_size) % self.window_size
#         x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
#         _, _, Hp, Wp = x.shape
#
#         x_windows = window_partition(x, self.window_size)
#         attn_windows = self.attn(x_windows)
#         x = window_reverse(attn_windows, self.window_size, Hp, Wp)
#
#         if pad_r > 0 or pad_b > 0:
#             x = x[:, :, :H, :W]
#
#         x = x.permute(0, 2, 3, 1).view(B, H * W, C)
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#
#
# class RSTB(nn.Module):
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
#                  img_size=128, patch_size=1, resi_connection='1conv', use_edge_attention=False):
#         super(RSTB, self).__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(
#                 dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads,
#                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_layer=norm_layer, use_checkpoint=use_checkpoint, use_edge_attention=use_edge_attention)
#             for i in range(depth)])
#
#         self.resi_connection = resi_connection
#         if resi_connection == '1conv':
#             self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             self.conv = nn.Sequential(
#                 nn.Conv2d(dim, dim // 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(dim // 4, dim, 3, 1, 1)
#             )
#
#     def forward(self, x):
#         B, L, C = x.shape
#         H, W = self.input_resolution
#         for blk in self.blocks:
#             x = blk(x)
#         x = x.transpose(1, 2).view(B, C, H, W)
#         x = self.conv(x)
#         x = x.view(B, C, H * W).transpose(1, 2)
#         return x
#
#
# class Upsample(nn.Module):
#     def __init__(self, scale, num_feat):
#         super(Upsample, self).__init__()
#         self.up = nn.Sequential(
#             nn.Conv2d(num_feat, num_feat * scale * scale, 3, 1, 1),
#             nn.PixelShuffle(scale)
#         )
#
#     def forward(self, x):
#         return self.up(x)
#
#
# class UpsampleOneStep(nn.Module):
#     def __init__(self, scale, embed_dim, num_out_ch, input_resolution):
#         super().__init__()
#         self.scale = scale
#         self.proj = nn.Conv2d(embed_dim, num_out_ch * scale * scale, 1)
#         self.pixel_shuffle = nn.PixelShuffle(scale)
#
#     def forward(self, x):
#         return self.pixel_shuffle(self.proj(x))
#
#
# # ===================== 主模型 =====================
# class EnhancedTripleInputSwinIR(nn.Module):
#     """改进的三输入融合SwinIR变体
#     已修改：illumination/velocity 为0时自动禁用对应分支
#     """
#
#     def set_alpha(self, alpha):
#         """设置先验特征的权重系数"""
#         self.alpha = alpha
#
#     def __init__(self, img_size=128, patch_size=1, in_chans=1,
#                  embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
#                  window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
#                  use_edge_attention=False, **kwargs):
#         super(EnhancedTripleInputSwinIR, self).__init__()
#
#         # 基本参数
#         num_in_ch = in_chans
#         num_out_ch = in_chans
#         num_feat = 64
#         self.img_range = img_range
#         self.mean = torch.zeros(1, 1, 1, 1)
#         self.upscale = upscale
#         self.upsampler = upsampler
#         self.window_size = window_size
#         self.alpha = 1.0  # 默认权重为1.0
#
#         #####################################################################################################
#         # 1, 三输入融合层
#         #####################################################################################################
#         # 主输入处理
#         self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
#
#         # 震源照明处理分支 - 使用更深的网络
#         self.illumination_processor = nn.Sequential(
#             nn.Conv2d(1, 128, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(64, embed_dim, 3, 1, 1)
#         )
#
#         # 偏移速度模型处理分支 - 使用更深的网络
#         self.velocity_processor = nn.Sequential(
#             nn.Conv2d(1, 128, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(64, embed_dim, 3, 1, 1)
#         )
#
#         # 使用增强的特征融合模块
#         self.feature_fusion = PhysicsInformedFeatureFusion(embed_dim)
#
#         #####################################################################################################
#         # 2, 深层特征提取
#         #####################################################################################################
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = embed_dim
#         self.mlp_ratio = mlp_ratio
#
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         self.patch_unembed = PatchUnEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = RSTB(dim=embed_dim,
#                          input_resolution=(patches_resolution[0],
#                                            patches_resolution[1]),
#                          depth=depths[i_layer],
#                          num_heads=num_heads[i_layer],
#                          window_size=window_size,
#                          mlp_ratio=self.mlp_ratio,
#                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                          drop=drop_rate, attn_drop=attn_drop_rate,
#                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                          norm_layer=norm_layer,
#                          downsample=None,
#                          use_checkpoint=use_checkpoint,
#                          img_size=img_size,
#                          patch_size=patch_size,
#                          resi_connection=resi_connection,
#                          use_edge_attention=use_edge_attention)
#             self.layers.append(layer)
#         self.norm = norm_layer(self.num_features)
#
#         if resi_connection == '1conv':
#             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
#
#         #####################################################################################################
#         # 3, 高质量图像重建
#         #####################################################################################################
#         if self.upsampler == 'pixelshuffle':
#             self.conv_before_upsample = nn.Sequential(
#                 nn.Conv2d(embed_dim, num_feat * 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True)
#             )
#             self.upsample = Upsample(upscale, num_feat * 4)
#             self.conv_last = nn.Sequential(
#                 nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             )
#         elif self.upsampler == 'pixelshuffledirect':
#             self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
#                                             (patches_resolution[0], patches_resolution[1]))
#         elif self.upsampler == 'nearest+conv':
#             assert self.upscale == 4, 'only support x4 now.'
#             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
#                                                       nn.LeakyReLU(inplace=True))
#             self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         else:
#             self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
#         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#         return x
#
#     def forward_features(self, x):
#         x_size = (x.shape[2], x.shape[3])
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#
#         for layer in self.layers:
#             x = layer(x)
#
#         x = self.norm(x)
#         x = self.patch_unembed(x, x_size)
#         return x
#
#     # 【修改4】前向传播自动判断先验是否为0
#     def forward(self, x, illumination=None, velocity=None):
#         H, W = x.shape[2:]
#         x = self.check_image_size(x)
#
#         self.mean = self.mean.type_as(x)
#         x = (x - self.mean) * self.img_range
#
#         # 处理主输入
#         x_main = self.conv_first(x)
#
#         # 照明分支：为None或全0 → 置全0特征
#         if illumination is not None and not torch.all(torch.abs(illumination) < 1e-8):
#             illumination = self.check_image_size(illumination)
#             illumination_features = self.illumination_processor(illumination)
#             if illumination_features.shape[2:] != x_main.shape[2:]:
#                 illumination_features = F.interpolate(illumination_features,
#                                                        size=x_main.shape[2:],
#                                                        mode='bilinear', align_corners=False)
#         else:
#             illumination_features = torch.zeros_like(x_main)
#
#         # 速度分支：为None或全0 → 置全0特征
#         if velocity is not None and not torch.all(torch.abs(velocity) < 1e-8):
#             velocity = self.check_image_size(velocity)
#             velocity_features = self.velocity_processor(velocity)
#             if velocity_features.shape[2:] != x_main.shape[2:]:
#                 velocity_features = F.interpolate(velocity_features,
#                                                    size=x_main.shape[2:],
#                                                    mode='bilinear', align_corners=False)
#         else:
#             velocity_features = torch.zeros_like(x_main)
#
#         # 自适应融合
#         final_fused = self.feature_fusion(x_main, illumination_features, velocity_features)
#
#         # 深层特征提取与重建
#         if self.upsampler == 'pixelshuffle':
#             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
#             x = self.conv_before_upsample(x)
#             x = self.conv_last(self.upsample(x))
#         elif self.upsampler == 'pixelshuffledirect':
#             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
#             x = self.upsample(x)
#         elif self.upsampler == 'nearest+conv':
#             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
#             x = self.conv_before_upsample(x)
#             x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.conv_last(self.lrelu(self.conv_hr(x)))
#         else:
#             x_first = final_fused
#             res = self.conv_after_body(self.forward_features(x_first)) + x_first
#             x = x_first + self.conv_last(res)
#
#         x = x / self.img_range + self.mean
#         return x[:, :, :H * self.upscale, :W * self.upscale]
#
#     def flops(self):
#         flops = 0
#         H, W = self.patches_resolution
#         flops += H * W * 3 * self.embed_dim * 9
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += H * W * 3 * self.embed_dim * self.embed_dim
#         flops += self.upsample.flops()
#         return flops
#
# if __name__ == '__main__':
#     upscale = 4
#     window_size = 8
#     height = (1024 // upscale // window_size + 1) * window_size
#     width = (720 // upscale // window_size + 1) * window_size
#     model = EnhancedTripleInputSwinIR(upscale=2, img_size=(height, width),
#                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6,6,6],
#                    embed_dim=180, num_heads=[6, 6, 6, 6,6,6], mlp_ratio=2, upsampler='pixelshuffledirect')
#     print(model)
#     print(height, width, model.flops() / 1e9)
#
#     x = torch.randn((1, 3, height, width))
#     x = model(x)
#     print(x.shape)




# # lsm-3-input-mse-ssim-1e
# # -----------------------------------------------------------------------------------
# # SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# # Originally Written by Ze Liu, Modified by Jingyun Liang.
# # -----------------------------------------------------------------------------------
#
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size
#
#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows
#
#
# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image
#
#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x
#
#
# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.
#
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """
#
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#
#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)
#
#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
#
#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops
#
#
# # class WindowAttention(nn.Module):
# #     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
# #                  attn_drop=0., proj_drop=0., window_size=7):
# #         super().__init__()
# #         self.num_heads = num_heads
# #         self.window_size = window_size
# #         head_dim = dim // num_heads
# #         self.scale = qk_scale or head_dim ** -0.5
# #
# #         # EffAttention的降维策略
# #         self.reduce = nn.Linear(dim, dim // 2, bias=qkv_bias)
# #         self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
# #         self.proj = nn.Linear(dim // 2, dim)
# #         self.attn_drop = nn.Dropout(attn_drop)
# #         self.proj_drop = nn.Dropout(proj_drop)
# #
# #         # 从WindowAttention引入相对位置偏置
# #         self.relative_position_bias_table = nn.Parameter(
# #             torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
# #
# #         # 相对位置索引计算
# #         coords = torch.arange(window_size)
# #         coords = torch.stack(torch.meshgrid([coords, coords]))
# #         coords_flatten = torch.flatten(coords, 1)
# #         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
# #         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
# #         relative_coords[:, :, 0] += window_size - 1
# #         relative_coords[:, :, 1] += window_size - 1
# #         relative_coords[:, :, 0] *= 2 * window_size - 1
# #         relative_position_index = relative_coords.sum(-1)
# #         self.register_buffer("relative_position_index", relative_position_index)
# #
# #         trunc_normal_(self.relative_position_bias_table, std=.02)
# #         self.softmax = nn.Softmax(dim=-1)
# #
# #     def forward(self, x, mask=None):
# #         # 降维
# #         x_reduced = self.reduce(x)
# #         B, N, C = x_reduced.shape
# #
# #         # QKV计算
# #         qkv = self.qkv(x_reduced).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
# #         q, k, v = qkv[0], qkv[1], qkv[2]
# #
# #         # EffAttention的分块计算
# #         chunk_size = max(1, N // 4)  # 动态分块
# #         q_chunks = torch.split(q, chunk_size, dim=-2)
# #         k_chunks = torch.split(k, chunk_size, dim=-2)
# #         v_chunks = torch.split(v, chunk_size, dim=-2)
# #
# #         outputs = []
# #         for i, (q_chunk, k_chunk, v_chunk) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
# #             # 注意力计算
# #             attn = (q_chunk @ k_chunk.transpose(-2, -1)) * self.scale
# #
# #             # 添加相对位置偏置（关键改进！）
# #             chunk_start = i * chunk_size
# #             chunk_end = min((i + 1) * chunk_size, N)
# #             chunk_relative_bias = self._get_chunk_relative_bias(chunk_start, chunk_end, N)
# #             attn = attn + chunk_relative_bias
# #
# #             attn = self.softmax(attn)
# #             attn = self.attn_drop(attn)
# #
# #             chunk_output = (attn @ v_chunk).transpose(1, 2)
# #             outputs.append(chunk_output)
# #
# #         # 合并结果
# #         x = torch.cat(outputs, dim=1)
# #         x = x.reshape(B, N, C)
# #         x = self.proj(x)
# #         x = self.proj_drop(x)
# #         return x
# #
# #     def _get_chunk_relative_bias(self, start, end, total_len):
# #         # 获取当前chunk的相对位置偏置
# #         indices = self.relative_position_index[start:end, start:end]
# #         relative_bias = self.relative_position_bias_table[indices.reshape(-1)]
# #         relative_bias = relative_bias.reshape(end - start, end - start, -1).permute(2, 0, 1)
# #         return relative_bias.unsqueeze(0)
#
# class EdgeAwareAttention(nn.Module):
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         # 原有的相对位置偏置表
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
#
#         # 边缘敏感的位置偏置（新增）
#         self.edge_bias = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
#
#         # 获取相对位置索引
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
#         coords_flatten = torch.flatten(coords, 1)
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#         relative_coords[:, :, 0] += self.window_size[0] - 1
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)
#         self.register_buffer("relative_position_index", relative_position_index)
#
#         # 原有的QKV和投影层
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         trunc_normal_(self.edge_bias, std=.02)  # 初始化边缘偏置
#         self.softmax = nn.Softmax(dim=-1)
#
#         # 边缘检测卷积核
#         self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                                                      dtype=torch.float32).view(1, 1, 3, 3))
#         self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                                                      dtype=torch.float32).view(1, 1, 3, 3))
#
#     def compute_edge_features(self, x):
#         """计算边缘特征"""
#         B, H, W, C = x.shape
#
#         # 处理任意通道数的输入
#         if C == 1:
#             # 单通道直接使用
#             gray = x.squeeze(-1)
#         elif C == 3:
#             # RGB转换为灰度
#             gray = 0.299 * x[:, :, :, 0] + 0.587 * x[:, :, :, 1] + 0.114 * x[:, :, :, 2]
#         else:
#             # 多通道特征图，取平均值作为灰度表示
#             gray = torch.mean(x, dim=-1)
#
#         # 添加批次和通道维度
#         gray = gray.reshape(B, 1, H, W)
#
#         # 计算梯度
#         grad_x = F.conv2d(gray, self.sobel_x.to(x.device), padding=1)
#         grad_y = F.conv2d(gray, self.sobel_y.to(x.device), padding=1)
#
#         # 计算梯度幅值
#         edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
#
#         # 调整形状
#         edge_strength = edge_strength.reshape(B, H, W, 1)
#
#         return edge_strength
#
#     def forward(self, x, mask=None, edge_enhance=True):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#             edge_enhance: 是否启用边缘增强
#         """
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#
#         # 计算相对位置偏置
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#
#         # 添加边缘感知偏置（如果启用）
#         if edge_enhance:
#             # 将窗口特征转换回空间格式
#             window_size = int(math.sqrt(N))
#             x_spatial = x.view(-1, window_size, window_size, C)
#
#             # 计算边缘特征
#             edge_features = self.compute_edge_features(x_spatial)
#
#             # 将边缘特征转换为注意力偏置
#             edge_bias = self.edge_bias[self.relative_position_index.view(-1)].view(
#                 self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
#             edge_bias = edge_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
#
#             # 使用边缘特征加权边缘偏置
#             edge_strength = edge_features.view(-1, window_size * window_size, 1)
#             edge_strength = edge_strength.mean(dim=1, keepdim=True)  # 平均边缘强度 [B_, 1, 1]
#
#             # 重新设计边缘偏置计算，确保维度匹配
#             # 将edge_bias扩展到与edge_strength相同的批次维度
#             expanded_edge_bias = edge_bias.unsqueeze(0)  # [1, num_heads, N, N]
#             expanded_edge_strength = edge_strength.unsqueeze(1)  # [B_, 1, 1, 1]
#
#             # 应用边缘强度加权
#             enhanced_edge_bias = expanded_edge_bias * expanded_edge_strength  # [B_, num_heads, N, N]
#
#             # 添加到注意力权重
#             attn = attn + relative_position_bias.unsqueeze(0) + enhanced_edge_bias
#         else:
#             attn = attn + relative_position_bias.unsqueeze(0)
#
#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
# class SwinTransformerBlock(nn.Module):
#     r""" Swin Transformer Block.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_edge_attention=False):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         self.use_edge_attention = use_edge_attention  # 新增参数
#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
#
#         self.norm1 = norm_layer(dim)
#
#         self.attn = WindowAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#
#         # # 修改为
#         # self.attn = WindowAttention(
#         #     dim, num_heads=num_heads, window_size=self.window_size,  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#
#
#         # # 修改这里：使用EdgeAwareAttention替代WindowAttention
#         # self.attn = EdgeAwareAttention(
#         #     dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#         #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#         if self.shift_size > 0:
#             attn_mask = self.calculate_mask(self.input_resolution)
#         else:
#             attn_mask = None
#
#         self.register_buffer("attn_mask", attn_mask)
#
#     def calculate_mask(self, x_size):
#         # calculate attention mask for SW-MSA
#         H, W = x_size
#         img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
#         h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1
#
#         mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#
#         return attn_mask
#
#     def forward(self, x, x_size):
#         H, W = x_size
#         B, L, C = x.shape
#         # assert L == H * W, "input feature has wrong size"
#
#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)
#
#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x
#
#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
#
#         # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
#         if self.input_resolution == x_size:
#             attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
#         else:
#             attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
#
#         # # W-MSA/SW-MSA
#         # # 修改这里：传递edge_enhance参数
#         # if self.input_resolution == x_size:
#         #     attn_windows = self.attn(x_windows, mask=self.attn_mask, edge_enhance=self.use_edge_attention)
#         # else:
#         #     attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device),
#         #                              edge_enhance=self.use_edge_attention)
#
#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
#
#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x
#         x = x.view(B, H * W, C)
#
#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
#
#     def flops(self):
#         flops = 0
#         H, W = self.input_resolution
#         # norm1
#         flops += self.dim * H * W
#         # W-MSA/SW-MSA
#         nW = H * W / self.window_size / self.window_size
#         flops += nW * self.attn.flops(self.window_size * self.window_size)
#         # mlp
#         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#         # norm2
#         flops += self.dim * H * W
#         return flops
#
#
# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#
#     Args:
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
#
#         x = x.view(B, H, W, C)
#
#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
#
#         x = self.norm(x)
#         x = self.reduction(x)
#
#         return x
#
#     def extra_repr(self) -> str:
#         return f"input_resolution={self.input_resolution}, dim={self.dim}"
#
#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.dim
#         flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
#         return flops
#
#
# class BasicLayer(nn.Module):
#     """ A basic Swin Transformer layer for one stage.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """
#
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, use_edge_attention=False):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#         self.use_edge_attention = use_edge_attention  # 新增参数
#
#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
#                                  num_heads=num_heads, window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                  drop=drop, attn_drop=attn_drop,
#                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                  norm_layer=norm_layer,
#                                  use_edge_attention=use_edge_attention)
#             for i in range(depth)])
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#
#     def forward(self, x, x_size):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x, x_size)
#             else:
#                 x = blk(x, x_size)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
#
#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops
#
#
# class RSTB(nn.Module):
#     """Residual Swin Transformer Block (RSTB).
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#         img_size: Input image size.
#         patch_size: Patch size.
#         resi_connection: The convolutional block before residual connection.
#     """
#
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
#                  img_size=224, patch_size=4, resi_connection='1conv',use_edge_attention=False):
#         super(RSTB, self).__init__()
#
#         self.dim = dim
#         self.input_resolution = input_resolution
#
#         self.residual_group = BasicLayer(dim=dim,
#                                          input_resolution=input_resolution,
#                                          depth=depth,
#                                          num_heads=num_heads,
#                                          window_size=window_size,
#                                          mlp_ratio=mlp_ratio,
#                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                          drop=drop, attn_drop=attn_drop,
#                                          drop_path=drop_path,
#                                          norm_layer=norm_layer,
#                                          downsample=downsample,
#                                          use_checkpoint=use_checkpoint,
#                                          use_edge_attention=use_edge_attention)
#
#         if resi_connection == '1conv':
#             self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             # to save parameters and memory
#             self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                       nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
#                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                       nn.Conv2d(dim // 4, dim, 3, 1, 1))
#
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
#             norm_layer=None)
#
#         self.patch_unembed = PatchUnEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
#             norm_layer=None)
#
#     def forward(self, x, x_size):
#         return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
#
#     def flops(self):
#         flops = 0
#         flops += self.residual_group.flops()
#         H, W = self.input_resolution
#         flops += H * W * self.dim * self.dim * 9
#         flops += self.patch_embed.flops()
#         flops += self.patch_unembed.flops()
#
#         return flops
#
#
# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding
#
#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """
#
#     def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]
#
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None
#
#     def forward(self, x):
#         x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
#         if self.norm is not None:
#             x = self.norm(x)
#         return x
#
#     def flops(self):
#         flops = 0
#         H, W = self.img_size
#         if self.norm is not None:
#             flops += H * W * self.embed_dim
#         return flops
#
#
# class PatchUnEmbed(nn.Module):
#     r""" Image to Patch Unembedding
#
#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """
#
#     def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]
#
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#
#     def forward(self, x, x_size):
#         B, HW, C = x.shape
#         x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
#         return x
#
#     def flops(self):
#         flops = 0
#         return flops
#
#
# class Upsample(nn.Sequential):
#     """Upsample module.
#
#     Args:
#         scale (int): Scale factor. Supported scales: 2^n and 3.
#         num_feat (int): Channel number of intermediate features.
#     """
#
#     def __init__(self, scale, num_feat):
#         m = []
#         if (scale & (scale - 1)) == 0:  # scale = 2^n
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
#                 m.append(nn.PixelShuffle(2))
#         elif scale == 3:
#             m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
#             m.append(nn.PixelShuffle(3))
#         else:
#             raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
#         super(Upsample, self).__init__(*m)
#
#
# class UpsampleOneStep(nn.Sequential):
#     """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
#        Used in lightweight SR to save parameters.
#
#     Args:
#         scale (int): Scale factor. Supported scales: 2^n and 3.
#         num_feat (int): Channel number of intermediate features.
#
#     """
#
#     def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
#         self.num_feat = num_feat
#         self.input_resolution = input_resolution
#         m = []
#         m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
#         m.append(nn.PixelShuffle(scale))
#         super(UpsampleOneStep, self).__init__(*m)
#
#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.num_feat * 3 * 9
#         return flops
#
#
# # class SwinIR(nn.Module):
# #     r""" SwinIR
# #         A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
# #
# #     Args:
# #         img_size (int | tuple(int)): Input image size. Default 64
# #         patch_size (int | tuple(int)): Patch size. Default: 1
# #         in_chans (int): Number of input image channels. Default: 3
# #         embed_dim (int): Patch embedding dimension. Default: 96
# #         depths (tuple(int)): Depth of each Swin Transformer layer.
# #         num_heads (tuple(int)): Number of attention heads in different layers.
# #         window_size (int): Window size. Default: 7
# #         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
# #         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
# #         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
# #         drop_rate (float): Dropout rate. Default: 0
# #         attn_drop_rate (float): Attention dropout rate. Default: 0
# #         drop_path_rate (float): Stochastic depth rate. Default: 0.1
# #         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
# #         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
# #         patch_norm (bool): If True, add normalization after patch embedding. Default: True
# #         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
# #         upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
# #         img_range: Image range. 1. or 255.
# #         upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
# #         resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
# #     """
# #
# #     def __init__(self, img_size=64, patch_size=1, in_chans=3,
# #                  embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
# #                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
# #                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
# #                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
# #                  use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
# #                  use_edge_attention=True,**kwargs):
# #         super(SwinIR, self).__init__()
# #         num_in_ch = in_chans
# #         num_out_ch = in_chans
# #         num_feat = 64
# #         self.img_range = img_range
# #         if in_chans == 3:
# #             rgb_mean = (0.4488, 0.4371, 0.4040)
# #             self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
# #         else:
# #             self.mean = torch.zeros(1, 1, 1, 1)
# #         self.upscale = upscale
# #         self.upsampler = upsampler
# #         self.window_size = window_size
# #
# #         #####################################################################################################
# #         ################################### 1, shallow feature extraction ###################################
# #         self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
# #
# #         #####################################################################################################
# #         ################################### 2, deep feature extraction ######################################
# #         self.num_layers = len(depths)
# #         self.embed_dim = embed_dim
# #         self.ape = ape
# #         self.patch_norm = patch_norm
# #         self.num_features = embed_dim
# #         self.mlp_ratio = mlp_ratio
# #
# #         # split image into non-overlapping patches
# #         self.patch_embed = PatchEmbed(
# #             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
# #             norm_layer=norm_layer if self.patch_norm else None)
# #         num_patches = self.patch_embed.num_patches
# #         patches_resolution = self.patch_embed.patches_resolution
# #         self.patches_resolution = patches_resolution
# #
# #         # merge non-overlapping patches into image
# #         self.patch_unembed = PatchUnEmbed(
# #             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
# #             norm_layer=norm_layer if self.patch_norm else None)
# #
# #         # absolute position embedding
# #         if self.ape:
# #             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
# #             trunc_normal_(self.absolute_pos_embed, std=.02)
# #
# #         self.pos_drop = nn.Dropout(p=drop_rate)
# #
# #         # stochastic depth
# #         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
# #
# #         # build Residual Swin Transformer blocks (RSTB)
# #         self.layers = nn.ModuleList()
# #         for i_layer in range(self.num_layers):
# #             layer = RSTB(dim=embed_dim,
# #                          input_resolution=(patches_resolution[0],
# #                                            patches_resolution[1]),
# #                          depth=depths[i_layer],
# #                          num_heads=num_heads[i_layer],
# #                          window_size=window_size,
# #                          mlp_ratio=self.mlp_ratio,
# #                          qkv_bias=qkv_bias, qk_scale=qk_scale,
# #                          drop=drop_rate, attn_drop=attn_drop_rate,
# #                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
# #                          norm_layer=norm_layer,
# #                          downsample=None,
# #                          use_checkpoint=use_checkpoint,
# #                          img_size=img_size,
# #                          patch_size=patch_size,
# #                          resi_connection=resi_connection,
# #                          use_edge_attention=use_edge_attention
# #
# #                          )
# #             self.layers.append(layer)
# #         self.norm = norm_layer(self.num_features)
# #
# #         # build the last conv layer in deep feature extraction
# #         if resi_connection == '1conv':
# #             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
# #         elif resi_connection == '3conv':
# #             # to save parameters and memory
# #             self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
# #                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #                                                  nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
# #                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #                                                  nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
# #
# #         #####################################################################################################
# #         ################################ 3, high quality image reconstruction ################################
# #         if self.upsampler == 'pixelshuffle':
# #             # for classical SR
# #             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
# #                                                       nn.LeakyReLU(inplace=True))
# #             self.upsample = Upsample(upscale, num_feat)
# #             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
# #         elif self.upsampler == 'pixelshuffledirect':
# #             # for lightweight SR (to save parameters)
# #             self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
# #                                             (patches_resolution[0], patches_resolution[1]))
# #         elif self.upsampler == 'nearest+conv':
# #             # for real-world SR (less artifacts)
# #             assert self.upscale == 4, 'only support x4 now.'
# #             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
# #                                                       nn.LeakyReLU(inplace=True))
# #             self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #             self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #             self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
# #             self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# #         else:
# #             # for image denoising and JPEG compression artifact reduction
# #             self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
# #
# #         self.apply(self._init_weights)
# #
# #     def _init_weights(self, m):
# #         if isinstance(m, nn.Linear):
# #             trunc_normal_(m.weight, std=.02)
# #             if isinstance(m, nn.Linear) and m.bias is not None:
# #                 nn.init.constant_(m.bias, 0)
# #         elif isinstance(m, nn.LayerNorm):
# #             nn.init.constant_(m.bias, 0)
# #             nn.init.constant_(m.weight, 1.0)
# #
# #     @torch.jit.ignore
# #     def no_weight_decay(self):
# #         return {'absolute_pos_embed'}
# #
# #     @torch.jit.ignore
# #     def no_weight_decay_keywords(self):
# #         return {'relative_position_bias_table'}
# #
# #     def check_image_size(self, x):
# #         _, _, h, w = x.size()
# #         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
# #         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
# #         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
# #         return x
# #
# #     def forward_features(self, x):
# #         x_size = (x.shape[2], x.shape[3])
# #         x = self.patch_embed(x)
# #         if self.ape:
# #             x = x + self.absolute_pos_embed
# #         x = self.pos_drop(x)
# #
# #         for layer in self.layers:
# #             x = layer(x, x_size)
# #
# #         x = self.norm(x)  # B L C
# #         x = self.patch_unembed(x, x_size)
# #
# #         return x
# #
# #     def forward(self, x):
# #         H, W = x.shape[2:]
# #         x = self.check_image_size(x)
# #
# #         self.mean = self.mean.type_as(x)
# #         x = (x - self.mean) * self.img_range
# #
# #         if self.upsampler == 'pixelshuffle':
# #             # for classical SR
# #             x = self.conv_first(x)
# #             x = self.conv_after_body(self.forward_features(x)) + x
# #             x = self.conv_before_upsample(x)
# #             x = self.conv_last(self.upsample(x))
# #         elif self.upsampler == 'pixelshuffledirect':
# #             # for lightweight SR
# #             x = self.conv_first(x)
# #             x = self.conv_after_body(self.forward_features(x)) + x
# #             x = self.upsample(x)
# #         elif self.upsampler == 'nearest+conv':
# #             # for real-world SR
# #             x = self.conv_first(x)
# #             x = self.conv_after_body(self.forward_features(x)) + x
# #             x = self.conv_before_upsample(x)
# #             x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
# #             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
# #             x = self.conv_last(self.lrelu(self.conv_hr(x)))
# #         else:
# #             # for image denoising and JPEG compression artifact reduction
# #             x_first = self.conv_first(x)
# #             res = self.conv_after_body(self.forward_features(x_first)) + x_first
# #             x = x + self.conv_last(res)
# #
# #         x = x / self.img_range + self.mean
# #
# #         return x[:, :, :H*self.upscale, :W*self.upscale]
# #
# #     def flops(self):
# #         flops = 0
# #         H, W = self.patches_resolution
# #         flops += H * W * 3 * self.embed_dim * 9
# #         flops += self.patch_embed.flops()
# #         for i, layer in enumerate(self.layers):
# #             flops += layer.flops()
# #         flops += H * W * 3 * self.embed_dim * self.embed_dim
# #         flops += self.upsample.flops()
# #         return flops
#
#
#
# # class TripleInputSwinIR(nn.Module):
# #     """支持三输入融合的SwinIR变体"""
# #
# #     def __init__(self, img_size=64, patch_size=1, in_chans=1,
# #                  embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
# #                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
# #                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
# #                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
# #                  use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
# #                  use_edge_attention=False, **kwargs):
# #         super(TripleInputSwinIR, self).__init__()
# #
# #         # 基本参数
# #         num_in_ch = in_chans
# #         num_out_ch = in_chans
# #         num_feat = 64
# #         self.img_range = img_range
# #         self.mean = torch.zeros(1, 1, 1, 1)  # 单通道数据的均值
# #         self.upscale = upscale
# #         self.upsampler = upsampler
# #         self.window_size = window_size
# #
# #         #####################################################################################################
# #         ################################### 1, 三输入融合层 ###################################
# #         # 主输入处理
# #         self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
# #
# #         # # 震源照明处理分支
# #         # self.illumination_processor = nn.Sequential(
# #         #     nn.Conv2d(1, 32, 3, 1, 1),
# #         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #         #     nn.Conv2d(32, 32, 3, 1, 1),
# #         #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
# #         # )
# #         #
# #         # # 偏移速度模型处理分支
# #         # self.velocity_processor = nn.Sequential(
# #         #     nn.Conv2d(1, 32, 3, 1, 1),
# #         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #         #     nn.Conv2d(32, 32, 3, 1, 1),
# #         #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
# #         # )
# #         #
# #         # # 特征融合层
# #         # self.fusion_conv = nn.Conv2d(embed_dim + 32 + 32, embed_dim, 3, 1, 1)
# #
# #         # 震源照明处理分支 - 使用残差块
# #         self.illumination_processor = nn.Sequential(
# #             nn.Conv2d(1, 32, 3, 1, 1),
# #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #             ResidualBlock(32),
# #             ResidualBlock(32)
# #         )
# #
# #         # 偏移速度模型处理分支 - 使用残差块
# #         self.velocity_processor = nn.Sequential(
# #             nn.Conv2d(1, 32, 3, 1, 1),
# #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #             ResidualBlock(32),
# #             ResidualBlock(32)
# #         )
# #
# #         # 特征融合层 - 使用注意力机制
# #         self.fusion_attention = nn.Sequential(
# #             nn.Conv2d(embed_dim + 32 + 32, embed_dim, 3, 1, 1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
# #             nn.Sigmoid()
# #         )
# #         self.fusion_conv = nn.Conv2d(embed_dim + 32 + 32, embed_dim, 3, 1, 1)
# #
# #         #####################################################################################################
# #         ################################### 2, 深層特征提取 ###################################
# #         self.num_layers = len(depths)
# #         self.embed_dim = embed_dim
# #         self.ape = ape
# #         self.patch_norm = patch_norm
# #         self.num_features = embed_dim
# #         self.mlp_ratio = mlp_ratio
# #
# #         # split image into non-overlapping patches
# #         self.patch_embed = PatchEmbed(
# #             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
# #             norm_layer=norm_layer if self.patch_norm else None)
# #         num_patches = self.patch_embed.num_patches
# #         patches_resolution = self.patch_embed.patches_resolution
# #         self.patches_resolution = patches_resolution
# #
# #         # merge non-overlapping patches into image
# #         self.patch_unembed = PatchUnEmbed(
# #             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
# #             norm_layer=norm_layer if self.patch_norm else None)
# #
# #         # absolute position embedding
# #         if self.ape:
# #             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
# #             trunc_normal_(self.absolute_pos_embed, std=.02)
# #
# #         self.pos_drop = nn.Dropout(p=drop_rate)
# #
# #         # stochastic depth
# #         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
# #
# #         # build Residual Swin Transformer blocks (RSTB)
# #         self.layers = nn.ModuleList()
# #         for i_layer in range(self.num_layers):
# #             layer = RSTB(dim=embed_dim,
# #                          input_resolution=(patches_resolution[0],
# #                                            patches_resolution[1]),
# #                          depth=depths[i_layer],
# #                          num_heads=num_heads[i_layer],
# #                          window_size=window_size,
# #                          mlp_ratio=self.mlp_ratio,
# #                          qkv_bias=qkv_bias, qk_scale=qk_scale,
# #                          drop=drop_rate, attn_drop=attn_drop_rate,
# #                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
# #                          norm_layer=norm_layer,
# #                          downsample=None,
# #                          use_checkpoint=use_checkpoint,
# #                          img_size=img_size,
# #                          patch_size=patch_size,
# #                          resi_connection=resi_connection,
# #                          use_edge_attention=use_edge_attention)
# #             self.layers.append(layer)
# #         self.norm = norm_layer(self.num_features)
# #
# #         # build the last conv layer in deep feature extraction
# #         if resi_connection == '1conv':
# #             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
# #         elif resi_connection == '3conv':
# #             # to save parameters and memory
# #             self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
# #                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #                                                  nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
# #                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #                                                  nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
# #
# #         #####################################################################################################
# #         ################################ 3, 高质量图像重建 ################################
# #         if self.upsampler == 'pixelshuffle':
# #             # for classical SR
# #             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
# #                                                       nn.LeakyReLU(inplace=True))
# #             self.upsample = Upsample(upscale, num_feat)
# #             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
# #         elif self.upsampler == 'pixelshuffledirect':
# #             # for lightweight SR (to save parameters)
# #             self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
# #                                             (patches_resolution[0], patches_resolution[1]))
# #         elif self.upsampler == 'nearest+conv':
# #             # for real-world SR (less artifacts)
# #             assert self.upscale == 4, 'only support x4 now.'
# #             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
# #                                                       nn.LeakyReLU(inplace=True))
# #             self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #             self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #             self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
# #             self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# #         else:
# #             # for image denoising and JPEG compression artifact reduction
# #             self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
# #
# #         self.apply(self._init_weights)
# #
# #     def _init_weights(self, m):
# #         if isinstance(m, nn.Linear):
# #             trunc_normal_(m.weight, std=.02)
# #             if isinstance(m, nn.Linear) and m.bias is not None:
# #                 nn.init.constant_(m.bias, 0)
# #         elif isinstance(m, nn.LayerNorm):
# #             nn.init.constant_(m.bias, 0)
# #             nn.init.constant_(m.weight, 1.0)
# #
# #     @torch.jit.ignore
# #     def no_weight_decay(self):
# #         return {'absolute_pos_embed'}
# #
# #     @torch.jit.ignore
# #     def no_weight_decay_keywords(self):
# #         return {'relative_position_bias_table'}
# #
# #     def check_image_size(self, x):
# #         _, _, h, w = x.size()
# #         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
# #         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
# #         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
# #         return x
# #
# #     def forward_features(self, x):
# #         x_size = (x.shape[2], x.shape[3])
# #         x = self.patch_embed(x)
# #         if self.ape:
# #             x = x + self.absolute_pos_embed
# #         x = self.pos_drop(x)
# #
# #         for layer in self.layers:
# #             x = layer(x, x_size)
# #
# #         x = self.norm(x)  # B L C
# #         x = self.patch_unembed(x, x_size)
# #
# #         return x
# #
# #     def forward(self, x, illumination=None, velocity=None):
# #         # 三输入融合前向传播
# #         H, W = x.shape[2:]
# #         x = self.check_image_size(x)
# #
# #         self.mean = self.mean.type_as(x)
# #         x = (x - self.mean) * self.img_range
# #
# #         # 处理震源照明输入
# #         if illumination is not None:
# #             illumination = self.check_image_size(illumination)
# #             illumination_features = self.illumination_processor(illumination)
# #         else:
# #             illumination_features = torch.zeros_like(x)
# #
# #         # 处理偏移速度模型输入
# #         if velocity is not None:
# #             velocity = self.check_image_size(velocity)
# #             velocity_features = self.velocity_processor(velocity)
# #         else:
# #             velocity_features = torch.zeros_like(x)
# #
# #         # 三输入融合
# #         x = self.conv_first(x)
# #         # 调整先验特征的尺寸（如果尺寸不匹配）
# #         if illumination_features.shape[2:] != x.shape[2:]:
# #             illumination_features = F.interpolate(illumination_features, size=x.shape[2:], mode='bilinear',
# #                                                   align_corners=False)
# #         if velocity_features.shape[2:] != x.shape[2:]:
# #             velocity_features = F.interpolate(velocity_features, size=x.shape[2:], mode='bilinear',
# #                                               align_corners=False)
# #         combined_features = torch.cat([x, illumination_features, velocity_features], dim=1)
# #         attention_weights = self.fusion_attention(combined_features)
# #         # fused_features = self.fusion_conv(combined_features)
# #         fused_features = attention_weights * self.fusion_conv(combined_features) + x  # 跳跃连接
# #
# #         # 深層特征提取
# #         if self.upsampler == 'pixelshuffle':
# #             # for classical SR
# #             x = self.conv_after_body(self.forward_features(fused_features)) + fused_features
# #             x = self.conv_before_upsample(x)
# #             x = self.conv_last(self.upsample(x))
# #         elif self.upsampler == 'pixelshuffledirect':
# #             # for lightweight SR
# #             x = self.conv_after_body(self.forward_features(fused_features)) + fused_features
# #             x = self.upsample(x)
# #         elif self.upsampler == 'nearest+conv':
# #             # for real-world SR
# #             x = self.conv_after_body(self.forward_features(fused_features)) + fused_features
# #             x = self.conv_before_upsample(x)
# #             x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
# #             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
# #             x = self.conv_last(self.lrelu(self.conv_hr(x)))
# #         else:
# #             # for image denoising and JPEG compression artifact reduction
# #             x_first = fused_features
# #             res = self.conv_after_body(self.forward_features(x_first)) + x_first
# #             x = x_first + self.conv_last(res)
# #
# #         x = x / self.img_range + self.mean
# #
# #         return x[:, :, :H * self.upscale, :W * self.upscale]
# #
# #     def flops(self):
# #         flops = 0
# #         H, W = self.patches_resolution
# #         flops += H * W * 3 * self.embed_dim * 9
# #         flops += self.patch_embed.flops()
# #         for i, layer in enumerate(self.layers):
# #             flops += layer.flops()
# #         flops += H * W * 3 * self.embed_dim * self.embed_dim
# #         flops += self.upsample.flops()
# #         return flops
# # # 残差块定义
# # class ResidualBlock(nn.Module):
# #     def __init__(self, n_feats):
# #         super(ResidualBlock, self).__init__()
# #         self.body = nn.Sequential(
# #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# #         )
# #         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# #
# #     def forward(self, x):
# #         res = self.body(x)
# #         res += x
# #         return self.act(res)
# # if __name__ == '__main__':
# #     upscale = 4
# #     window_size = 8
# #     height = (1024 // upscale // window_size + 1) * window_size
# #     width = (720 // upscale // window_size + 1) * window_size
# #     model = TripleInputSwinIR(upscale=2, img_size=(height, width),
# #                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
# #                    embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
# #     print(model)
# #     print(height, width, model.flops() / 1e9)
# #
# #     x = torch.randn((1, 3, height, width))
# #     x = model(x)
# #     print(x.shape)
#
# # superresolution-1-enhance-3-l1
# # class PhysicsInformedFeatureFusion(nn.Module):
# #     """基于地球物理原理的特征融合模块"""
# #
# #     def __init__(self, embed_dim):
# #         super(PhysicsInformedFeatureFusion, self).__init__()
# #         self.embed_dim = embed_dim
# #
# #         # 照明增强模块 - 模拟地震照明效应
# #         self.illumination_enhancer = nn.Sequential(
# #             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
# #             nn.LeakyReLU(0.2, True),
# #             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
# #             nn.Sigmoid()  # 输出在0-1之间，作为权重
# #         )
# #
# #         # 速度约束模块 - 模拟速度对波传播的影响
# #         self.velocity_constraint = nn.Sequential(
# #             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
# #             nn.LeakyReLU(0.2, True),
# #             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
# #         )
# #
# #         # 多尺度特征提取 - 捕捉不同尺度的地质特征
# #         self.multi_scale_extract = nn.ModuleList([
# #             nn.Sequential(
# #                 nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
# #                 nn.LeakyReLU(0.2, True),
# #                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, 1, 1)
# #             ) for _ in range(3)
# #         ])
# #
# #         # 多尺度特征融合
# #         self.scale_fusion = nn.Sequential(
# #             nn.Conv2d(embed_dim + embed_dim // 4 * 3, embed_dim, 1, 1, 0),
# #             nn.LeakyReLU(0.2, True),
# #             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
# #         )
# #
# #         # 物理约束注意力机制
# #         self.physics_attention = nn.Sequential(
# #             nn.Conv2d(embed_dim * 3, embed_dim, 3, 1, 1),
# #             nn.LeakyReLU(0.2, True),
# #             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
# #             nn.Sigmoid()
# #         )
# #
# #         # 最终融合
# #         self.final_fusion = nn.Sequential(
# #             nn.Conv2d(embed_dim * 3, embed_dim * 2, 3, 1, 1),
# #             nn.LeakyReLU(0.2, True),
# #             nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)
# #         )
# #
# #     def forward(self, x_main, illum_features, velocity_features):
# #         # 应用照明增强 - 模拟地震照明效应
# #         illumination_weights = self.illumination_enhancer(illum_features)
# #         x_illuminated = x_main * illumination_weights
# #
# #         # 应用速度约束 - 模拟速度场对波传播的影响
# #         velocity_constraint = self.velocity_constraint(velocity_features)
# #         x_velocity_constrained = x_illuminated + velocity_constraint
# #
# #         # 多尺度特征提取 - 捕捉不同尺度的地质特征
# #         scale_features = []
# #         for i, extractor in enumerate(self.multi_scale_extract):
# #             scale_factor = 2 ** (i + 1)
# #             scaled = F.interpolate(x_velocity_constrained, scale_factor=1 / scale_factor,
# #                                    mode='bilinear', align_corners=False)
# #             feat = extractor(scaled)
# #             feat = F.interpolate(feat, size=x_velocity_constrained.shape[2:],
# #                                  mode='bilinear', align_corners=False)
# #             scale_features.append(feat)
# #
# #         # 融合多尺度特征
# #         multi_scale_feat = torch.cat([x_velocity_constrained] + scale_features, dim=1)
# #         x_multi_scale = self.scale_fusion(multi_scale_feat)
# #
# #         # 物理约束注意力机制
# #         attention_input = torch.cat([x_multi_scale, illum_features, velocity_features], dim=1)
# #         attention_weights = self.physics_attention(attention_input)
# #
# #         # 应用注意力权重
# #         attended_features = x_multi_scale * attention_weights
# #
# #         # 最终融合
# #         final_combined = torch.cat([attended_features, illum_features, velocity_features], dim=1)
# #         output = self.final_fusion(final_combined) + x_main  # 残差连接
# #
# #         return output
# #
# #
# # class EnhancedTripleInputSwinIR(nn.Module):
# #     """改进的三输入融合SwinIR变体 - 结合乘法融合和注意力重加权"""
# #
# #     def __init__(self, img_size=128, patch_size=1, in_chans=1,
# #                  embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
# #                  window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
# #                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
# #                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
# #                  use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
# #                  use_edge_attention=False, **kwargs):
# #         super(EnhancedTripleInputSwinIR, self).__init__()
# #
# #         # 基本参数
# #         num_in_ch = in_chans
# #         num_out_ch = in_chans
# #         num_feat = 64
# #         self.img_range = img_range
# #         self.mean = torch.zeros(1, 1, 1, 1)
# #         self.upscale = upscale
# #         self.upsampler = upsampler
# #         self.window_size = window_size
# #
# #         #####################################################################################################
# #         ################################### 1, 三输入融合层 ###################################
# #         # 主输入处理
# #         self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
# #
# #         # 震源照明处理分支 - 使用更深的网络
# #         self.illumination_processor = nn.Sequential(
# #             nn.Conv2d(1, 128, 3, 1, 1),
# #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #             ResidualBlock(128),
# #             ResidualBlock(128),
# #             ResidualBlock(128),
# #             nn.Conv2d(128, 64, 3, 1, 1),
# #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #             nn.Conv2d(64, embed_dim, 3, 1, 1)  # 直接输出与主特征相同的维度
# #         )
# #
# #         # 偏移速度模型处理分支 - 使用更深的网络
# #         self.velocity_processor = nn.Sequential(
# #             nn.Conv2d(1, 128, 3, 1, 1),
# #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #             ResidualBlock(128),
# #             ResidualBlock(128),
# #             ResidualBlock(128),
# #             nn.Conv2d(128, 64, 3, 1, 1),
# #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #             nn.Conv2d(64, embed_dim, 3, 1, 1)  # 直接输出与主特征相同的维度
# #         )
# #
# #
# #         # 使用增强的特征融合模块
# #         self.feature_fusion = PhysicsInformedFeatureFusion(embed_dim)
# #
# #         #####################################################################################################
# #         ################################### 2, 深層特征提取 ###################################
# #         self.num_layers = len(depths)
# #         self.embed_dim = embed_dim
# #         self.ape = ape
# #         self.patch_norm = patch_norm
# #         self.num_features = embed_dim
# #         self.mlp_ratio = mlp_ratio
# #
# #         # split image into non-overlapping patches
# #         self.patch_embed = PatchEmbed(
# #             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
# #             norm_layer=norm_layer if self.patch_norm else None)
# #         num_patches = self.patch_embed.num_patches
# #         patches_resolution = self.patch_embed.patches_resolution
# #         self.patches_resolution = patches_resolution
# #
# #         # merge non-overlapping patches into image
# #         self.patch_unembed = PatchUnEmbed(
# #             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
# #             norm_layer=norm_layer if self.patch_norm else None)
# #
# #         # absolute position embedding
# #         if self.ape:
# #             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
# #             trunc_normal_(self.absolute_pos_embed, std=.02)
# #
# #         self.pos_drop = nn.Dropout(p=drop_rate)
# #
# #         # stochastic depth
# #         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
# #
# #         # build Residual Swin Transformer blocks (RSTB)
# #         self.layers = nn.ModuleList()
# #         for i_layer in range(self.num_layers):
# #             layer = RSTB(dim=embed_dim,
# #                          input_resolution=(patches_resolution[0],
# #                                            patches_resolution[1]),
# #                          depth=depths[i_layer],
# #                          num_heads=num_heads[i_layer],
# #                          window_size=window_size,
# #                          mlp_ratio=self.mlp_ratio,
# #                          qkv_bias=qkv_bias, qk_scale=qk_scale,
# #                          drop=drop_rate, attn_drop=attn_drop_rate,
# #                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
# #                          norm_layer=norm_layer,
# #                          downsample=None,
# #                          use_checkpoint=use_checkpoint,
# #                          img_size=img_size,
# #                          patch_size=patch_size,
# #                          resi_connection=resi_connection,
# #                          use_edge_attention=use_edge_attention)
# #             self.layers.append(layer)
# #         self.norm = norm_layer(self.num_features)
# #
# #         # build the last conv layer in deep feature extraction
# #         if resi_connection == '1conv':
# #             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
# #         elif resi_connection == '3conv':
# #             self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
# #                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #                                                  nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
# #                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #                                                  nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
# #
# #         #####################################################################################################
# #         ################################ 3, 高质量图像重建 ################################
# #         if self.upsampler == 'pixelshuffle':
# #             # self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
# #             #                                           nn.LeakyReLU(inplace=True))
# #             # self.upsample = Upsample(upscale, num_feat)
# #             # self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
# #             # 增强的上采样路径
# #             self.conv_before_upsample = nn.Sequential(
# #                 nn.Conv2d(embed_dim, num_feat * 4, 3, 1, 1),
# #                 nn.LeakyReLU(0.2, True),
# #                 nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
# #                 nn.LeakyReLU(0.2, True),
# #                 nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
# #                 nn.LeakyReLU(0.2, True)
# #             )
# #             self.upsample = Upsample(upscale, num_feat * 4)
# #             self.conv_last = nn.Sequential(
# #                 nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1),
# #                 nn.LeakyReLU(0.2, True),
# #                 nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
# #                 nn.LeakyReLU(0.2, True),
# #                 nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
# #             )
# #
# #         elif self.upsampler == 'pixelshuffledirect':
# #             self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
# #                                             (patches_resolution[0], patches_resolution[1]))
# #         elif self.upsampler == 'nearest+conv':
# #             assert self.upscale == 4, 'only support x4 now.'
# #             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
# #                                                       nn.LeakyReLU(inplace=True))
# #             self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #             self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #             self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
# #             self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# #         else:
# #             self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
# #
# #         self.apply(self._init_weights)
# #
# #     def _init_weights(self, m):
# #         if isinstance(m, nn.Linear):
# #             trunc_normal_(m.weight, std=.02)
# #             if isinstance(m, nn.Linear) and m.bias is not None:
# #                 nn.init.constant_(m.bias, 0)
# #         elif isinstance(m, nn.LayerNorm):
# #             nn.init.constant_(m.bias, 0)
# #             nn.init.constant_(m.weight, 1.0)
# #
# #     @torch.jit.ignore
# #     def no_weight_decay(self):
# #         return {'absolute_pos_embed'}
# #
# #     @torch.jit.ignore
# #     def no_weight_decay_keywords(self):
# #         return {'relative_position_bias_table'}
# #
# #     def check_image_size(self, x):
# #         _, _, h, w = x.size()
# #         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
# #         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
# #         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
# #         return x
# #
# #     def forward_features(self, x):
# #         x_size = (x.shape[2], x.shape[3])
# #         x = self.patch_embed(x)
# #         if self.ape:
# #             x = x + self.absolute_pos_embed
# #         x = self.pos_drop(x)
# #
# #         for layer in self.layers:
# #             x = layer(x, x_size)
# #
# #         x = self.norm(x)
# #         x = self.patch_unembed(x, x_size)
# #
# #         return x
# #
# #     def forward(self, x, illumination=None, velocity=None):
# #         # 三输入融合前向传播
# #         H, W = x.shape[2:]
# #         x = self.check_image_size(x)
# #
# #         self.mean = self.mean.type_as(x)
# #         x = (x - self.mean) * self.img_range
# #
# #         # 处理主输入
# #         x_main = self.conv_first(x)
# #
# #         # 处理震源照明输入 (方案一: 乘法融合)
# #         if illumination is not None:
# #             illumination = self.check_image_size(illumination)
# #             # # 归一化以提高数值稳定性
# #             # illumination = (illumination - torch.mean(illumination)) / (torch.std(illumination) + 1e-8)
# #             illumination_features = self.illumination_processor(illumination)
# #             # # 调整通道数
# #             # illumination_features_adjusted = self.illumination_channel_adjust(illumination_features)
# #             # 调整尺寸
# #             if illumination_features.shape[2:] != x_main.shape[2:]:
# #                 illumination_features_adjusted = F.interpolate(illumination_features,
# #                                                                size=x_main.shape[2:],
# #                                                                mode='bilinear', align_corners=False)
# #             # # 乘法融合: 主特征 × 照明特征
# #             # x_illuminated = x_main * illumination_features_adjusted
# #         else:
# #             illumination_features = torch.ones_like(x_main)  # 使用1而不是0
# #             # x_illuminated = x_main
# #
# #
# #         # 处理偏移速度模型输入
# #         if velocity is not None:
# #             velocity = self.check_image_size(velocity)
# #             # velocity = (velocity - torch.mean(velocity)) / (torch.std(velocity) + 1e-8)
# #             velocity_features = self.velocity_processor(velocity)
# #             # # 调整通道数
# #             # velocity_features_adjusted = self.velocity_channel_adjust(velocity_features)
# #             # 调整尺寸
# #             if velocity_features.shape[2:] != x_main.shape[2:]:
# #                 velocity_features_adjusted = F.interpolate(velocity_features,
# #                                                            size=x_main.shape[2:],
# #                                                            mode='bilinear', align_corners=False)
# #         else:
# #             velocity_features = torch.zeros_like(x_main)
# #
# #         # # 使用增强的特征融合模块 (符合GAN做法: 相乘结果与偏移速度模型拼接)
# #         final_fused = self.feature_fusion(x_main, illumination_features, velocity_features)
# #
# #         # 深層特征提取
# #         if self.upsampler == 'pixelshuffle':
# #             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
# #             x = self.conv_before_upsample(x)
# #             x = self.conv_last(self.upsample(x))
# #         elif self.upsampler == 'pixelshuffledirect':
# #             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
# #             x = self.upsample(x)
# #         elif self.upsampler == 'nearest+conv':
# #             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
# #             x = self.conv_before_upsample(x)
# #             x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
# #             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
# #             x = self.conv_last(self.lrelu(self.conv_hr(x)))
# #         else:
# #             x_first = final_fused
# #             res = self.conv_after_body(self.forward_features(x_first)) + x_first
# #             x = x_first + self.conv_last(res)
# #
# #         x = x / self.img_range + self.mean
# #
# #         return x[:, :, :H * self.upscale, :W * self.upscale]
# #
# #     def flops(self):
# #         flops = 0
# #         H, W = self.patches_resolution
# #         flops += H * W * 3 * self.embed_dim * 9
# #         flops += self.patch_embed.flops()
# #         for i, layer in enumerate(self.layers):
# #             flops += layer.flops()
# #         flops += H * W * 3 * self.embed_dim * self.embed_dim
# #         flops += self.upsample.flops()
# #         return flops
# #
# #
# # # # 残差块定义
# # # class ResidualBlock(nn.Module):
# # #     def __init__(self, n_feats):
# # #         super(ResidualBlock, self).__init__()
# # #         self.body = nn.Sequential(
# # #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# # #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# # #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# # #         )
# # #         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# # #
# # #     def forward(self, x):
# # #         res = self.body(x)
# # #         res += x
# # #         return self.act(res)
# #
# # # 增强的残差块
# # class ResidualBlock(nn.Module):
# #     def __init__(self, n_feats, reduction=16):
# #         super(ResidualBlock, self).__init__()
# #         self.body = nn.Sequential(
# #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# #             nn.LeakyReLU(0.2, True),
# #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# #         )
# #
# #         # 通道注意力
# #         self.ca = nn.Sequential(
# #             nn.AdaptiveAvgPool2d(1),
# #             nn.Conv2d(n_feats, n_feats // reduction, 1, 1, 0),
# #             nn.ReLU(True),
# #             nn.Conv2d(n_feats // reduction, n_feats, 1, 1, 0),
# #             nn.Sigmoid()
# #         )
# #
# #         # 空间注意力
# #         self.sa = nn.Sequential(
# #             nn.Conv2d(2, 1, 7, 1, 3),
# #             nn.Sigmoid()
# #         )
# #
# #         self.act = nn.LeakyReLU(0.2, True)
# #
# #     def forward(self, x):
# #         res = self.body(x)
# #
# #         # 通道注意力
# #         ca_weights = self.ca(res)
# #         res = res * ca_weights
# #
# #         # 空间注意力
# #         avg_out = torch.mean(res, dim=1, keepdim=True)
# #         max_out, _ = torch.max(res, dim=1, keepdim=True)
# #         sa_weights = self.sa(torch.cat([avg_out, max_out], dim=1))
# #         res = res * sa_weights
# #
# #         res += x
# #         return self.act(res)
# #
# # if __name__ == '__main__':
# #     upscale = 4
# #     window_size = 8
# #     height = (1024 // upscale // window_size + 1) * window_size
# #     width = (720 // upscale // window_size + 1) * window_size
# #     model = EnhancedTripleInputSwinIR(upscale=2, img_size=(height, width),
# #                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
# #                    embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
# #     print(model)
# #     print(height, width, model.flops() / 1e9)
# #
# #     x = torch.randn((1, 3, height, width))
# #     x = model(x)
# #     print(x.shape)
#
#
#
#
#
# class PhysicsInformedFeatureFusion(nn.Module):
#     """基于地球物理原理的特征融合模块"""
#
#     def __init__(self, embed_dim):
#         super(PhysicsInformedFeatureFusion, self).__init__()
#         self.embed_dim = embed_dim
#         # 添加监控属性
#         self.avg_illum_gate = 0.0
#         self.avg_vel_gate = 0.0
#         # 照明增强模块 - 模拟地震照明效应
#         self.illumination_enhancer = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.Sigmoid()  # 输出在0-1之间，作为权重
#         )
#
#         # 速度约束模块 - 模拟速度对波传播的影响
#         self.velocity_constraint = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#         )
#
#
#         # 多尺度特征提取 - 捕捉不同尺度的地质特征
#         self.multi_scale_extract = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, 1, 1)
#             ) for _ in range(3)
#         ])
#
#         # 多尺度特征融合
#         self.scale_fusion = nn.Sequential(
#             nn.Conv2d(embed_dim + embed_dim // 4 * 3, embed_dim, 1, 1, 0),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         )
#
#         # 物理约束注意力机制
#         self.physics_attention = nn.Sequential(
#             nn.Conv2d(embed_dim * 3, embed_dim, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
#             nn.Sigmoid()
#         )
#
#         # 最终融合
#         self.final_fusion = nn.Sequential(
#             nn.Conv2d(embed_dim * 3, embed_dim * 2, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)
#         )
#
#
#
#     def forward(self, x_main, illum_features, velocity_features):
#         # 保存原始输入用于残差连接
#         identity = x_main
#         # 应用照明增强 - 模拟地震照明效应
#         illumination_weights = self.illumination_enhancer(illum_features)
#         x_illuminated = x_main * illumination_weights
#
#         # 应用速度约束 - 模拟速度场对波传播的影响
#         velocity_constraint = self.velocity_constraint(velocity_features)
#         x_velocity_constrained = x_illuminated + velocity_constraint
#
#
#         # 多尺度特征提取 - 捕捉不同尺度的地质特征
#         scale_features = []
#         for i, extractor in enumerate(self.multi_scale_extract):
#             scale_factor = 2 ** (i + 1)
#             scaled = F.interpolate(x_velocity_constrained, scale_factor=1 / scale_factor,
#                                    mode='bilinear', align_corners=False)
#             feat = extractor(scaled)
#             feat = F.interpolate(feat, size=x_velocity_constrained.shape[2:],
#                                  mode='bilinear', align_corners=False)
#             scale_features.append(feat)
#
#         # 融合多尺度特征
#         multi_scale_feat = torch.cat([x_velocity_constrained] + scale_features, dim=1)
#         x_multi_scale = self.scale_fusion(multi_scale_feat)
#
#         # 物理约束注意力机制
#         attention_input = torch.cat([x_multi_scale, illum_features, velocity_features], dim=1)
#         attention_weights = self.physics_attention(attention_input)
#
#         # 应用注意力权重
#         attended_features = x_multi_scale * attention_weights
#
#         # 最终融合
#         final_combined = torch.cat([attended_features, illum_features, velocity_features], dim=1)
#         output = self.final_fusion(final_combined)
#         # 强有力的残差连接：最终输出必须包含原始的 x_main
#         output = output + identity
#
#         return output
#
#
# class EnhancedTripleInputSwinIR(nn.Module):
#     """改进的三输入融合SwinIR变体 - 结合乘法融合和注意力重加权"""
#
#     def set_alpha(self, alpha):
#         """设置先验特征的权重系数"""
#         self.alpha = alpha
#
#     def __init__(self, img_size=128, patch_size=1, in_chans=1,
#                  embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
#                  window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
#                  use_edge_attention=False, **kwargs):
#         super(EnhancedTripleInputSwinIR, self).__init__()
#
#         # 基本参数
#         num_in_ch = in_chans
#         num_out_ch = in_chans
#         num_feat = 64
#         self.img_range = img_range
#         self.mean = torch.zeros(1, 1, 1, 1)
#         self.upscale = upscale
#         self.upsampler = upsampler
#         self.window_size = window_size
#         self.alpha = 1.0  # 默认权重为1.0
#         #####################################################################################################
#         ################################### 1, 三输入融合层 ###################################
#         # 主输入处理
#         self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
#
#         # 震源照明处理分支 - 使用更深的网络
#         self.illumination_processor = nn.Sequential(
#             nn.Conv2d(1, 128, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(64, embed_dim, 3, 1, 1)  # 直接输出与主特征相同的维度
#         )
#
#         # 偏移速度模型处理分支 - 使用更深的网络
#         self.velocity_processor = nn.Sequential(
#             nn.Conv2d(1, 128, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             ResidualBlock(128),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(64, embed_dim, 3, 1, 1)  # 直接输出与主特征相同的维度
#         )
#
#         # # 简化的先验信息处理器
#         # self.illumination_processor = nn.Sequential(
#         #     nn.Conv2d(1, 64, 3, 1, 1),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         #     nn.Conv2d(64, embed_dim, 3, 1, 1)
#         # )
#         #
#         # self.velocity_processor = nn.Sequential(
#         #     nn.Conv2d(1, 64, 3, 1, 1),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         #     nn.Conv2d(64, embed_dim, 3, 1, 1)
#         # )
#
#         # 使用增强的特征融合模块
#         self.feature_fusion = PhysicsInformedFeatureFusion(embed_dim)
#
#         #####################################################################################################
#         ################################### 2, 深層特征提取 ###################################
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = embed_dim
#         self.mlp_ratio = mlp_ratio
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         # merge non-overlapping patches into image
#         self.patch_unembed = PatchUnEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#
#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#
#         # build Residual Swin Transformer blocks (RSTB)
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = RSTB(dim=embed_dim,
#                          input_resolution=(patches_resolution[0],
#                                            patches_resolution[1]),
#                          depth=depths[i_layer],
#                          num_heads=num_heads[i_layer],
#                          window_size=window_size,
#                          mlp_ratio=self.mlp_ratio,
#                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                          drop=drop_rate, attn_drop=attn_drop_rate,
#                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                          norm_layer=norm_layer,
#                          downsample=None,
#                          use_checkpoint=use_checkpoint,
#                          img_size=img_size,
#                          patch_size=patch_size,
#                          resi_connection=resi_connection,
#                          use_edge_attention=use_edge_attention)
#             self.layers.append(layer)
#         self.norm = norm_layer(self.num_features)
#
#         # build the last conv layer in deep feature extraction
#         if resi_connection == '1conv':
#             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
#
#         #####################################################################################################
#         ################################ 3, 高质量图像重建 ################################
#         if self.upsampler == 'pixelshuffle':
#             # self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
#             #                                           nn.LeakyReLU(inplace=True))
#             # self.upsample = Upsample(upscale, num_feat)
#             # self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             # 增强的上采样路径
#             self.conv_before_upsample = nn.Sequential(
#                 nn.Conv2d(embed_dim, num_feat * 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True)
#             )
#             self.upsample = Upsample(upscale, num_feat * 4)
#             self.conv_last = nn.Sequential(
#                 nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             )
#
#         elif self.upsampler == 'pixelshuffledirect':
#             self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
#                                             (patches_resolution[0], patches_resolution[1]))
#         elif self.upsampler == 'nearest+conv':
#             assert self.upscale == 4, 'only support x4 now.'
#             self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
#                                                       nn.LeakyReLU(inplace=True))
#             self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         else:
#             self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
#         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#         return x
#
#     def forward_features(self, x):
#         x_size = (x.shape[2], x.shape[3])
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#
#         for layer in self.layers:
#             x = layer(x, x_size)
#
#         x = self.norm(x)
#         x = self.patch_unembed(x, x_size)
#
#         return x
#
#     def forward(self, x, illumination=None, velocity=None):
#         # 三输入融合前向传播
#         H, W = x.shape[2:]
#         x = self.check_image_size(x)
#
#         self.mean = self.mean.type_as(x)
#         x = (x - self.mean) * self.img_range
#
#         # 处理主输入
#         x_main = self.conv_first(x)
#
#         # 处理震源照明输入 (方案一: 乘法融合)
#         if illumination is not None:
#             illumination = self.check_image_size(illumination)
#             # # 归一化以提高数值稳定性
#             # illumination = (illumination - torch.mean(illumination)) / (torch.std(illumination) + 1e-8)
#             illumination_features = self.illumination_processor(illumination)
#             # # 调整通道数
#             # illumination_features_adjusted = self.illumination_channel_adjust(illumination_features)
#             # # 应用渐进式训练的alpha系数
#             # illumination_features = illumination_features * self.alpha
#             # 调整尺寸
#             if illumination_features.shape[2:] != x_main.shape[2:]:
#                 illumination_features_adjusted = F.interpolate(illumination_features,
#                                                                size=x_main.shape[2:],
#                                                                mode='bilinear', align_corners=False)
#             # # 乘法融合: 主特征 × 照明特征
#             # x_illuminated = x_main * illumination_features_adjusted
#         else:
#             illumination_features = torch.ones_like(x_main)*self.alpha  # 使用1而不是0
#             # x_illuminated = x_main
#
#
#         # 处理偏移速度模型输入
#         if velocity is not None:
#             velocity = self.check_image_size(velocity)
#             # velocity = (velocity - torch.mean(velocity)) / (torch.std(velocity) + 1e-8)
#             velocity_features = self.velocity_processor(velocity)
#             # # 调整通道数
#             # velocity_features_adjusted = self.velocity_channel_adjust(velocity_features)
#             # # 应用渐进式训练的alpha系数
#             # velocity_features = velocity_features * self.alpha
#             # 调整尺寸
#             if velocity_features.shape[2:] != x_main.shape[2:]:
#                 velocity_features_adjusted = F.interpolate(velocity_features,
#                                                            size=x_main.shape[2:],
#                                                            mode='bilinear', align_corners=False)
#         else:
#             velocity_features = torch.zeros_like(x_main)
#
#         # # 使用增强的特征融合模块 (符合GAN做法: 相乘结果与偏移速度模型拼接)
#         final_fused = self.feature_fusion(x_main, illumination_features, velocity_features)
#
#         # 深層特征提取
#         if self.upsampler == 'pixelshuffle':
#
#             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
#             x = self.conv_before_upsample(x)
#             x = self.conv_last(self.upsample(x))
#         elif self.upsampler == 'pixelshuffledirect':
#             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
#             x = self.upsample(x)
#         elif self.upsampler == 'nearest+conv':
#             x = self.conv_after_body(self.forward_features(final_fused)) + final_fused
#             x = self.conv_before_upsample(x)
#             x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
#             x = self.conv_last(self.lrelu(self.conv_hr(x)))
#         else:
#             x_first = final_fused
#             res = self.conv_after_body(self.forward_features(x_first)) + x_first
#             x = x_first + self.conv_last(res)
#
#         x = x / self.img_range + self.mean
#
#         return x[:, :, :H * self.upscale, :W * self.upscale]
#
#     def flops(self):
#         flops = 0
#         H, W = self.patches_resolution
#         flops += H * W * 3 * self.embed_dim * 9
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += H * W * 3 * self.embed_dim * self.embed_dim
#         flops += self.upsample.flops()
#         return flops
#
#
# # # 残差块定义
# # class ResidualBlock(nn.Module):
# #     def __init__(self, n_feats):
# #         super(ResidualBlock, self).__init__()
# #         self.body = nn.Sequential(
# #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
# #         )
# #         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
# #
# #     def forward(self, x):
# #         res = self.body(x)
# #         res += x
# #         return self.act(res)
#
# # 增强的残差块
# class ResidualBlock(nn.Module):
#     def __init__(self, n_feats, reduction=16):
#         super(ResidualBlock, self).__init__()
#         self.body = nn.Sequential(
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(n_feats, n_feats, 3, 1, 1),
#         )
#
#         # 通道注意力
#         self.ca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(n_feats, n_feats // reduction, 1, 1, 0),
#             nn.ReLU(True),
#             nn.Conv2d(n_feats // reduction, n_feats, 1, 1, 0),
#             nn.Sigmoid()
#         )
#
#         # 空间注意力
#         self.sa = nn.Sequential(
#             nn.Conv2d(2, 1, 7, 1, 3),
#             nn.Sigmoid()
#         )
#
#         self.act = nn.LeakyReLU(0.2, True)
#
#     def forward(self, x):
#         res = self.body(x)
#
#         # 通道注意力
#         ca_weights = self.ca(res)
#         res = res * ca_weights
#
#         # 空间注意力
#         avg_out = torch.mean(res, dim=1, keepdim=True)
#         max_out, _ = torch.max(res, dim=1, keepdim=True)
#         sa_weights = self.sa(torch.cat([avg_out, max_out], dim=1))
#         res = res * sa_weights
#
#         res += x
#         return self.act(res)
#
# if __name__ == '__main__':
#     upscale = 4
#     window_size = 8
#     height = (1024 // upscale // window_size + 1) * window_size
#     width = (720 // upscale // window_size + 1) * window_size
#     model = EnhancedTripleInputSwinIR(upscale=2, img_size=(height, width),
#                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6,6,6],
#                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
#     print(model)
#     print(height, width, model.flops() / 1e9)
#
#     x = torch.randn((1, 3, height, width))
#     x = model(x)
#     print(x.shape)
