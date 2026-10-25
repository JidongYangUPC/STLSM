# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# from models3.network_swinir import TripleInputSwinIR
from models3.loss_ssim import SSIMLoss
from models3.loss import MultiScalePerceptualLoss
from models3.loss import CombinedMSESSIMLoss
from models3.network_swinir import EnhancedTripleInputSwinIR
from models3.model_plain import ModelPlain


class ModelPlain3(ModelPlain):
    """三输入SwinIR模型，接受低分辨率图像、震源照明和偏移速度模型作为输入"""

    def __init__(self, opt):
        # 先调用父类的初始化
        super(ModelPlain3, self).__init__(opt)
        self.opt = opt

        # 重新定义网络以支持三输入
        self.netG = self.define_network(opt)
        self.netG = self.model_to_device(self.netG)

        # 如果使用指数移动平均，需要重新定义netE
        if self.opt_train['E_decay'] > 0:
            self.netE = self.define_network(opt).to(self.device).eval()
        # 关键：重新初始化损失函数
        # self.define_loss(opt)  # 调用父类的损失函数初始化方法
        # # 根据您的图像通道数设置channel参数
        # # 对于灰度图像，channel=1；对于RGB图像，channel=3
        # self.criterion = CombinedMSESSIMLoss(alpha=0.6, window_size=11, channel=1)
        # # 添加损失函数定义
        # # 使用L1损失作为主要损失函数
        # self.criterion = nn.L1Loss()

        # # 初始化损失函数
        # self.define_loss(opt)  # 这会设置 self.G_lossfn 和 self.G_lossfn_weight
        # # 添加多尺度感知损失
        # if opt['train'].get('G_lossfn_type') == 'multiscale_perceptual':
        #     self.criterion = MultiScalePerceptualLoss(
        #         scales=opt['train'].get('perceptual_scales', [1, 0.5, 0.25]),
        #         weights=opt['train'].get('perceptual_weights', [1.0, 0.5, 0.25]),
        #         grad_weight=opt['train'].get('grad_weight', 0.5)
        #     )
        # else:
        #     # 保持原有的损失函数
        #     self.criterion = self.init_loss(opt)

        # # 如果使用多尺度感知损失，覆盖父类的损失函数
        # if opt['train'].get('G_lossfn_type') == 'multiscale_perceptual':
        #     # 获取多尺度感知损失的参数
        #     scales = opt['train'].get('perceptual_scales', [1, 0.5, 0.25])
        #     weights = opt['train'].get('perceptual_weights', [1.0, 0.5, 0.25])
        #     grad_weight = opt['train'].get('grad_weight', 0.5)
        #
        #     # 创建多尺度感知损失函数
        #     self.G_lossfn = MultiScalePerceptualLoss(
        #         scales=scales,
        #         weights=weights,
        #         grad_weight=grad_weight
        #     ).to(self.device)
        #
        # # 确保 criterion 指向 G_lossfn，因为训练脚本中使用的是 model.criterion
        # self.criterion = self.G_lossfn
        # 验证损失函数
        self.validate_loss_function()

    def validate_loss_function(self):
        """验证损失函数是否正确配置"""
        if hasattr(self, 'G_lossfn'):
            print(f"使用的损失函数: {type(self.G_lossfn).__name__}")

            # 如果是复合损失函数，检查其组成
            if hasattr(self.G_lossfn, 'alpha'):
                print(f"SSIM权重(alpha): {self.G_lossfn.alpha}")
            if hasattr(self.G_lossfn, 'mse_loss'):
                print("包含MSE损失组件")
            if hasattr(self.G_lossfn, 'ssim_loss'):
                print("包含SSIM损失组件")
        else:
            print("警告: 未找到损失函数")

        # # 如果使用指数移动平均，需要重新定义netE
        # if self.opt_train['E_decay'] > 0:
        #     self.netE = self.define_network(opt).to(self.device).eval()

    def define_network(self, opt):
        # 创建三输入SwinIR网络
        return EnhancedTripleInputSwinIR(
            img_size=opt.get('img_size', 128),
            patch_size=opt.get('patch_size', 1),
            in_chans=opt.get('in_nc', 1),
            embed_dim=opt.get('embed_dim', 180),
            depths=opt.get('depths', [6, 6, 6, 6, 6, 6]),
            num_heads=opt.get('num_heads', [6, 6, 6, 6, 6, 6]),
            window_size=opt.get('window_size', 8),
            mlp_ratio=opt.get('mlp_ratio', 2),
            upscale=opt.get('scale', 2),
            img_range=opt.get('img_range', 1.0),
            upsampler=opt.get('upsampler', 'pixelshuffle'),
            resi_connection=opt.get('resi_connection', '1conv'),
            use_edge_attention=opt.get('use_edge_attention', False)
        )


    def feed_data(self, data, need_H=True):
        # 处理三输入数据
        self.L = data['L'].to(self.device)
        self.illumination = data['illumination'].to(self.device)  # 震源照明数据
        self.velocity = data['velocity'].to(self.device)  # 偏移速度模型数据
        if need_H:
            self.H = data['H'].to(self.device)

    def netG_forward(self):
        # 前向传播，使用三输入
        self.E = self.netG(self.L, self.illumination, self.velocity)
        return self.E  # 返回输出，便于在训练脚本中使用

    def test(self):
        # 测试时使用三输入
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()


