import torch
import math
import torch.nn as nn
import torchvision
import torch
from torch.nn import functional as F
from torch import autograd as autograd
from math import exp


"""
Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2*): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7*): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16*): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace)
      (25*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
      (26): ReLU(inplace)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace)
      (34*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
"""


# --------------------------------------------
# Perceptual loss
# --------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2,7,16,25,34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer)-1):
                self.features.add_module('child'+str(i), nn.Sequential(*list(model.features.children())[(feature_layer[i]+1):(feature_layer[i+1]+1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """

    def __init__(self, feature_layer=[2,7,16,25,34], weights=[0.1,0.1,1.0,1.0,1.0], lossfn_type='l1', use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss

# --------------------------------------------
# GAN loss: gan, ragan
# --------------------------------------------
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        elif self.gan_type == 'softplusgan':
            def softplusgan_loss(input, target):
                # target is boolean
                return F.softplus(-input).mean() if target else F.softplus(input).mean()

            self.loss = softplusgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan', 'softplusgan']:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss



def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.
        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.
    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.
    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


# --------------------------------------------
# Enhanced Charbonnier Loss with Edge Awareness
# --------------------------------------------
class EnhancedCharbonnierLoss(nn.Module):
    """
    Enhanced Charbonnier loss with edge-aware gradient regularization.
    This loss function improves edge reconstruction without adding extra modules.
    """

    def __init__(self, eps=1e-6, edge_weight=0.1, use_edge_mask=False,
                 mask_edge_weight=2.0, mask_dilation_size=3,reduction='mean'):
        super(EnhancedCharbonnierLoss, self).__init__()
        self.eps = eps
        self.edge_weight = edge_weight
        self.use_edge_mask = use_edge_mask
        self.mask_edge_weight = mask_edge_weight
        self.mask_dilation_size = mask_dilation_size
        self.reduction = reduction

    def forward(self, pred, target):
        # 基础Charbonnier损失
        base_loss = torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        # 如果需要使用边缘掩码
        if self.use_edge_mask:
            weight_mask = create_edge_aware_mask(
                target,
                dilation_size=self.mask_dilation_size,
                edge_threshold=0.1,
                base_weight=1.0,
                edge_weight=self.mask_edge_weight
            )
            base_loss = base_loss * weight_mask

        # 计算梯度差异（轻量级边缘感知）
        # 水平梯度差异
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_loss_x = torch.sqrt((pred_grad_x - target_grad_x) ** 2 + self.eps ** 2)

        # 垂直梯度差异
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss_y = torch.sqrt((pred_grad_y - target_grad_y) ** 2 + self.eps ** 2)

        # 组合梯度损失
        grad_loss = (grad_loss_x.mean() + grad_loss_y.mean()) * self.edge_weight

        # 组合总损失
        if self.reduction == 'mean':
            total_loss = base_loss.mean() + grad_loss
        elif self.reduction == 'sum':
            total_loss = base_loss.sum() + grad_loss
        else:
            total_loss = base_loss + grad_loss

        return total_loss


# --------------------------------------------
# Edge-aware mask generation
# --------------------------------------------
def create_edge_aware_mask(img, dilation_size=3, edge_threshold=0.1, base_weight=1.0, edge_weight=2.0):
    """
    Create edge-aware weight mask for focusing training on edge regions.

    Args:
        img (Tensor): Input image tensor [B, C, H, W]
        dilation_size (int): Size of dilation kernel to expand edge regions
        edge_threshold (float): Threshold for edge detection
        base_weight (float): Weight for non-edge regions
        edge_weight (float): Weight for edge regions

    Returns:
        Tensor: Weight mask with same shape as input
    """
    # 确保输入至少有4个维度
    if img.dim() == 3:
        img = img.unsqueeze(0)

    # 转换为灰度
    if img.size(1) == 3:  # RGB图像
        gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    else:  # 单通道图像
        gray = img.squeeze(1)

    # 添加通道维度
    gray = gray.unsqueeze(1)

    # 创建Sobel滤波器
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

    # 计算梯度
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # 检测边缘
    edges = gradient_magnitude > edge_threshold

    # 应用膨胀以扩大边缘区域
    if dilation_size > 1:
        kernel = torch.ones((1, 1, dilation_size, dilation_size), device=img.device)
        dilated_edges = F.conv2d(edges.float(), kernel, padding=dilation_size // 2) > 0
    else:
        dilated_edges = edges

    # 创建权重掩码
    weight_mask = torch.ones_like(img) * base_weight
    weight_mask[dilated_edges.expand_as(img)] = edge_weight

    return weight_mask


# --------------------------------------------
# Edge-aware weighted loss
# --------------------------------------------
class EdgeAwareWeightedLoss(nn.Module):
    """
    Edge-aware weighted loss that focuses training on edge regions.
    Can wrap any base loss function.
    """

    def __init__(self, base_loss_fn, edge_weight=2.0, dilation_size=3, edge_threshold=0.1):
        super(EdgeAwareWeightedLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.edge_weight = edge_weight
        self.dilation_size = dilation_size
        self.edge_threshold = edge_threshold

    def forward(self, pred, target):
        # 创建边缘感知权重掩码
        weight_mask = create_edge_aware_mask(
            target,
            dilation_size=self.dilation_size,
            edge_threshold=self.edge_threshold,
            base_weight=1.0,
            edge_weight=self.edge_weight
        )

        # 计算逐像素损失
        pixel_loss = torch.sqrt((pred - target) ** 2 + 1e-6)

        # 应用权重掩码
        weighted_loss = pixel_loss * weight_mask

        # 返回平均损失
        return weighted_loss.mean()


class MultiScalePerceptualLoss(nn.Module):
    def __init__(self, scales=[1, 0.5, 0.25], weights=[1.0, 0.5, 0.25], grad_weight=0.5):
        super(MultiScalePerceptualLoss, self).__init__()
        self.scales = scales
        self.weights = weights
        self.grad_weight = grad_weight  # 梯度损失权重
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        loss = 0
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1:
                pred_scaled = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                pred_scaled = pred
                target_scaled = target

            # 计算多尺度L1损失
            loss += weight * self.l1_loss(pred_scaled, target_scaled)

            # 计算梯度损失（保持边缘）
            pred_grad_x = torch.abs(pred_scaled[:, :, :, 1:] - pred_scaled[:, :, :, :-1])
            pred_grad_y = torch.abs(pred_scaled[:, :, 1:, :] - pred_scaled[:, :, :-1, :])
            target_grad_x = torch.abs(target_scaled[:, :, :, 1:] - target_scaled[:, :, :, :-1])
            target_grad_y = torch.abs(target_scaled[:, :, 1:, :] - target_scaled[:, :, :-1, :])

            loss += weight * self.grad_weight * (self.l1_loss(pred_grad_x, target_grad_x) +
                                    self.l1_loss(pred_grad_y, target_grad_y))

        return loss


class SSIMLoss(nn.Module):
    """
       支持任意输入范围的SSIM损失函数
       """
    """
    Structural Similarity Index (SSIM) loss
    Based on: https://github.com/Po-Hsun-Su/pytorch-ssim
    """

    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.data_range = 2.0
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        # # 自动检测数据范围
        # if self.data_range == 'auto':
        #     data_range = max(img1.max().item() - img1.min().item(),
        #                      img2.max().item() - img2.min().item())
        # else:
        #     data_range = self.data_range
        # Ensure the window is on the same device as input
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        # 根据数据范围调整常数
        data_range = self.data_range
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        # C1 = 0.01 ** 2
        # C2 = 0.03 ** 2
        # # 根据数据范围调整常数
        # data_range = self.data_range
        # C1 = (0.01 * data_range) ** 2
        # C2 = (0.03 * data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class CombinedMSESSIMLoss(nn.Module):
    """
    Combined MSE and SSIM loss as described in the paper
    L_new = α * L_SSIM + (1 - α) * L_MSE
    """

    def __init__(self, alpha=0.7, window_size=11, channel=1):
        super(CombinedMSESSIMLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss( window_size=window_size, channel=channel)

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.alpha * ssim + (1 - self.alpha) * mse

class CombinedL1SSIMLoss(nn.Module):
    """
    Combined MSE and SSIM loss as described in the paper
    L_new = α * L_SSIM + (1 - α) * L_1
    """

    def __init__(self, alpha=0.5, window_size=11, channel=1):
        super(CombinedL1SSIMLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss( window_size=window_size, channel=channel)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.alpha * ssim + (1 - self.alpha) * l1
