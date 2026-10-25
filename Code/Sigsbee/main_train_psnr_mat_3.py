# # -*-coding:utf-8-*-
# # 设置 matplotlib 使用非交互式后端
# import matplotlib
# matplotlib.use('Agg')  # 在导入任何 matplotlib 模块之前设置
# import os.path
# import math
# import argparse
# import random
# import numpy as np
# import logging
# import scipy.io as sio  # 添加MAT文件支持
# import matplotlib.pyplot as plt  # 添加绘图支持
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# import torch
#
# from utils import utils_logger
# from utils import utils_image as util
# from utils import utils_option as option
# from utils.utils_dist import get_dist_info, init_dist
#
# from data.select_dataset import define_Dataset
# from models3.select_model import define_Model
# from models3.model_plain3 import ModelPlain3
# # 添加混合精度训练和学习率调度的导入
# from torch.cuda.amp import autocast, GradScaler
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 或 '3'
# os.environ['MPLBACKEND'] = 'Agg'  # 设置环境变量确保使用 Agg 后端
# # 添加地震数据专用函数
# def save_seismic_data(data, save_path):
#     """保存地震数据为MAT文件"""
#     sio.savemat(save_path, {'sr': data})
#
#
# # 添加学习率热身函数
# # def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
# #     """创建带有线性热身的学习率调度器"""
# #
# #     def lr_lambda(current_step):
# #         if current_step < num_warmup_steps:
# #             return float(current_step) / float(max(1, num_warmup_steps))
# #         return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
# #
# #     return LambdaLR(optimizer, lr_lambda, last_epoch)
#
# def calculate_psnr_float(gt, pred, data_range=None):
#     """计算浮点数据的PSNR"""
#     mse = np.mean((gt - pred) ** 2)
#     if data_range is None:
#         data_range = gt.max() - gt.min()
#     if mse == 0:
#         return float('inf')
#     return 20 * np.log10(data_range / np.sqrt(mse))
#
#
# def calculate_ssim_float(gt, pred, data_range=None):
#     """计算浮点数据的SSIM"""
#     from skimage.metrics import structural_similarity as ssim
#     if data_range is None:
#         data_range = gt.max() - gt.min()
#     return ssim(gt, pred, data_range=data_range)
#
#
# '''
# # --------------------------------------------
# # training code for seismic data processing
# # --------------------------------------------
# '''
#
#
# def main(json_path='options/train_msrresnet_psnr.json'):
#     '''
#     # ----------------------------------------
#     # Step--1 (prepare opt)
#     # ----------------------------------------
#     '''
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
#     parser.add_argument('--launcher', default='pytorch', help='job launcher')
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument('--dist', default=False)
#
#     opt = option.parse(parser.parse_args().opt, is_train=True)
#     opt['dist'] = parser.parse_args().dist
#
#     # ----------------------------------------
#     # distributed settings
#     # ----------------------------------------
#     if opt['dist']:
#         init_dist('pytorch')
#     opt['rank'], opt['world_size'] = get_dist_info()
#
#     if opt['rank'] == 0:
#         util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
#
#     # ----------------------------------------
#     # update opt
#     # ----------------------------------------
#     init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
#     init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
#     opt['path']['pretrained_netG'] = init_path_G
#     opt['path']['pretrained_netE'] = init_path_E
#     init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
#                                                                              net_type='optimizerG')
#     opt['path']['pretrained_optimizerG'] = init_path_optimizerG
#     current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)
#
#     # 地震数据不需要border设置
#     # border = opt.get('scale', 1)  # 注释掉或删除
#
#     # ----------------------------------------
#     # save opt to  a '../option.json' file
#     # ----------------------------------------
#     if opt['rank'] == 0:
#         option.save(opt)
#
#     # ----------------------------------------
#     # return None for missing key
#     # ----------------------------------------
#     opt = option.dict_to_nonedict(opt)
#
#     # ----------------------------------------
#     # configure logger
#     # ----------------------------------------
#     if opt['rank'] == 0:
#         logger_name = 'train'
#         utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
#         logger = logging.getLogger(logger_name)
#         logger.info(option.dict2str(opt))
#
#     # ----------------------------------------
#     # seed
#     # ----------------------------------------
#     seed = opt['train']['manual_seed']
#     if seed is None:
#         seed = random.randint(1, 10000)
#     print('Random seed: {}'.format(seed))
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#     '''
#     # ----------------------------------------
#     # Step--2 (creat dataloader)
#     # ----------------------------------------
#     '''
#
#     # ----------------------------------------
#     # 1) create_dataset
#     # 2) creat_dataloader for train and test
#     # ----------------------------------------
#     for phase, dataset_opt in opt['datasets'].items():
#         if phase == 'train':
#             train_set = define_Dataset(dataset_opt)
#             train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
#             if opt['rank'] == 0:
#                 logger.info('Number of train samples: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
#             if opt['dist']:
#                 train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
#                                                    seed=seed)
#                 train_loader = DataLoader(train_set,
#                                           batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
#                                           shuffle=False,
#                                           num_workers=0,
#                                           drop_last=True,
#                                           pin_memory=False,
#                                           sampler=train_sampler)
#             else:
#                 train_loader = DataLoader(train_set,
#                                           batch_size=dataset_opt['dataloader_batch_size'],
#                                           shuffle=dataset_opt['dataloader_shuffle'],
#                                           num_workers=0,
#                                           drop_last=True,
#                                           pin_memory=False)
#
#         elif phase == 'test':
#             test_set = define_Dataset(dataset_opt)
#             test_loader = DataLoader(test_set, batch_size=1,
#                                      shuffle=False, num_workers=0,
#                                      drop_last=False, pin_memory=False)
#         else:
#             raise NotImplementedError("Phase [%s] is not recognized." % phase)
#
#     '''
#     # ----------------------------------------
#     # Step--3 (initialize model)
#     # ----------------------------------------
#     '''
#
#     # 确保模型配置适配单通道地震数据
#     if 'network_G' in opt and 'in_nc' in opt['network_G']:
#         opt['network_G']['in_nc'] = 1  # 单通道输入
#         opt['network_G']['out_nc'] = 1  # 单通道输出
#
#     # model = define_Model(opt)
#     model = ModelPlain3(opt)
#     model.init_train()
#
#     # 添加优化策略设置
#     # 1. 初始化混合精度训练的GradScaler
#     scaler = GradScaler()
#
#     # 2. 设置梯度累积步数
#     gradient_accumulation_steps = 4  # 根据您的GPU内存调整
#
#
#     # 3.禁用模型内部的学习率调度，使用我们自己的调度
#     #     # 但保留模型内部的优化器设置
#     model.schedulers = []
#
#     if opt['rank'] == 0:
#         logger.info(model.info_network())
#         logger.info(model.info_params())
#
#     '''
#     # ----------------------------------------
#     # Step--4 (main training)
#     # ----------------------------------------
#     '''
#
#     for epoch in range(800):  # 从计算出的轮数开始
#
#         if opt['dist']:
#             train_sampler.set_epoch(epoch + seed)
#
#         # # 添加梯度累积计数器
#         accumulation_steps = 0
#
#         for i, train_data in enumerate(train_loader):
#
#             current_step += 1
#             accumulation_steps += 1
#             # -------------------------------
#             # 1) update learning rate
#             # -------------------------------
#             # model.update_learning_rate(current_step)
#
#             # -------------------------------
#             # 2) feed patch pairs
#             # -------------------------------
#             model.feed_data(train_data)
#
#             # -------------------------------
#             # 3) optimize parameters
#             # -------------------------------
#             # model.optimize_parameters(current_step)
#             # 使用混合精度训练
#             with autocast():
#                 model.netG_forward()  # 前向传播
#                 loss = model.criterion(model.E, model.H)  # 计算损失
#                 loss = loss / gradient_accumulation_steps  # 梯度累积归一化
#
#             # 反向传播
#             scaler.scale(loss).backward()
#
#             # 梯度累积: 每accumulation_steps步更新一次参数
#             if accumulation_steps % gradient_accumulation_steps == 0:
#                 # 梯度裁剪
#                 scaler.unscale_(model.G_optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.netG.parameters(), max_norm=1.0)
#
#                 # 更新参数
#                 scaler.step(model.G_optimizer)
#                 scaler.update()
#
#                 # 清零梯度
#                 model.G_optimizer.zero_grad()
#
#                 # 更新学习率 - 使用模型内部的学习率调度
#                 model.update_learning_rate(current_step)
#
#                 # 重置累积计数器
#                 accumulation_steps = 0
#
#             # -------------------------------
#             # 4) training information
#             # -------------------------------
#             if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
#                 logs = model.current_log()  # such as loss
#                 # logs = {
#                 #    'loss': loss.item() * gradient_accumulation_steps
#                 #
#                 # }  # 恢复原始损失值
#                 message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch+1, current_step,
#                                                                           model.current_learning_rate())
#                 for k, v in logs.items():  # merge log information into message
#                     message += '{:s}: {:.3e} '.format(k, v)
#                 logger.info(message)
#
#                 # logs = model.current_log()  # such as loss
#                 # message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
#                 #                                                           model.current_learning_rate())
#                 # for k, v in logs.items():  # merge log information into message
#                 #     message += '{:s}: {:.3e} '.format(k, v)
#                 # logger.info(message)
#
#             # -------------------------------
#             # 5) save model
#             # -------------------------------
#             if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
#                 logger.info('Saving the model.')
#                 model.save(current_step)
#
#             # -------------------------------
#             # 6) testing and visualization
#             # -------------------------------
#             if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
#
#                 avg_psnr = 0.0
#                 avg_ssim = 0.0
#                 idx = 0
#
#                 for test_data in test_loader:
#                     idx += 1
#                     # 获取数据路径
#                     data_path_l = test_data['L_path'][0]
#                     data_path_illumination = test_data['illumination_path'][0]
#                     data_path_velocity = test_data['velocity_path'][0]
#
#                     # 使用低分辨率图像的名称作为主名称，或者创建组合名称
#                     data_name = os.path.splitext(os.path.basename(data_path_l))[0]
#
#                     # 创建保存目录
#                     data_dir = os.path.join(opt['path']['images'], data_name, f"epoch_{((current_step)//305):03d}")
#                     os.makedirs(data_dir, exist_ok=True)
#
#                     # 处理数据
#                     model.feed_data(test_data)
#                     model.test()
#
#                     # 获取结果
#                     visuals = model.current_visuals()
#
#                     # 地震数据不需要转换为uint8
#                     # 直接使用浮点张量
#                     E_tensor = visuals['E'].detach().cpu().float()
#                     H_tensor = visuals['H'].detach().cpu().float()
#                     input_tensor = test_data['L'].detach().cpu().float()
#                     # 转换为numpy数组
#                     E_data = E_tensor.squeeze().numpy()  # 移除批次和通道维度
#                     H_data = H_tensor.squeeze().numpy()
#                     input_data = input_tensor.squeeze().numpy()
#
#
#                     # -----------------------
#                     # 保存为MAT文件
#                     # -----------------------
#                     save_mat_path = os.path.join(data_dir, f'{data_name}.mat')
#                     save_seismic_data(E_data, save_mat_path)
#
#                     # -----------------------
#                     # 计算地震数据质量指标
#                     # -----------------------
#                     data_range = H_data.max() - H_data.min()
#                     current_psnr = calculate_psnr_float(H_data, E_data, data_range)
#                     current_ssim = calculate_ssim_float(H_data, E_data, data_range)
#
#                     logger.info('{:->4d}--> {:>10s} | PSNR: {:<4.2f}dB | SSIM: {:<4.4f}'.format(
#                         idx, os.path.basename(data_path_l), current_psnr, current_ssim))
#
#                     avg_psnr += current_psnr
#                     avg_ssim += current_ssim
#
#                     # -----------------------
#                     # 创建图片展示
#                     # -----------------------
#                     fig, axs = plt.subplots(1, 3, figsize=(6, 2))
#
#                     # # # 设置统一的颜色范围
#                     # vmin = min(E_data.min(), input_data.min(), H_data.min())
#                     # vmax = max(E_data.max(), input_data.max(), H_data.max())
#                     # vmin = -0.3
#                     # vmax = 0.3
#
#                     # # 绘制输入图像
#                     # im0 = axs[0].imshow(input_data, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
#                     # axs[0].set_title('Migrated')
#                     #
#                     #
#                     # # 绘制输出图像
#                     # im1 = axs[1].imshow(E_data, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
#                     # axs[1].set_title(f'Predicted\nPSNR: {current_psnr:.2f} dB\nSSIM: {current_ssim:.4f}')
#                     #
#                     #
#                     # # 绘制目标图像
#                     # im2 = axs[2].imshow(H_data, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
#                     # axs[2].set_title('Reflectivity')
#
#                     # 绘制输入图像
#                     im0 = axs[0].imshow(input_data, cmap=plt.cm.gray, vmin=-0.3, vmax=0.3)
#
#
#                     # 绘制输出图像
#                     im1 = axs[1].imshow(E_data, cmap=plt.cm.gray, vmin=-0.7, vmax=0.7)
#
#
#                     # 绘制目标图像
#                     im2 = axs[2].imshow(H_data, cmap=plt.cm.gray, vmin=-0.7, vmax=0.7)
#                     # 移除所有子图的刻度线和标签
#                     for ax in axs:
#                         ax.set_xticks([])  # 移除x轴刻度
#                         ax.set_yticks([])  # 移除y轴刻度
#                         ax.set_xticklabels([])  # 移除x轴标签
#                         ax.set_yticklabels([])  # 移除y轴标签
#
#                     # 保存图片
#                     plt.tight_layout()
#                     fig.savefig(os.path.join(data_dir, f'{data_name}.png'),bbox_inches='tight')
#                     plt.close(fig)
#                     plt.close('all')
#                     # # 测试循环结束后，恢复训练时使用的alpha值
#                     # if hasattr(model.netG, 'module'):
#                     #     model.netG.module.set_alpha(current_alpha)
#                     # else:
#                     #     model.netG.set_alpha(current_alpha)
#
#
#                 avg_psnr = avg_psnr / idx
#                 avg_ssim = avg_ssim / idx
#
#                 # # testing log
#                 # logger.info('<epoch:{:3d}, iter:{:8,d}, Avg PSNR: {:<.2f}dB, Avg SSIM: {:<.4f}\n'.format(
#                 #     epoch+1, current_step, avg_psnr, avg_ssim))
#
#                 # testing log
#                 # current_lr = scheduler.get_last_lr()[0] if scheduler else model.current_learning_rate()
#                 logger.info('<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, Avg PSNR: {:<.2f}dB, Avg SSIM: {:<.4f}\n'.format(
#                     epoch+1, current_step, model.current_learning_rate(), avg_psnr, avg_ssim))
#                 # # testing log
#                 # logger.info('<epoch:{:3d}, iter:{:8,d}, Avg PSNR: {:<.2f}dB, Avg SSIM: {:<.4f}\n'.format(
#                 #     epoch + 1, current_step, avg_psnr, avg_ssim))
#
#
# if __name__ == '__main__':
#     main()

# -*-coding:utf-8-*-
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import os.path
import math
import argparse
import random
import numpy as np
import logging
import scipy.io as sio  # 添加MAT文件支持
import matplotlib.pyplot as plt  # 添加绘图支持
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models3.select_model import define_Model
from models3.model_plain3 import ModelPlain3
# import matplotlib.pyplot as plt
from collections import defaultdict
import json
import time
# import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 或 '3'
os.environ['MPLBACKEND'] = 'Agg'  # 设置环境变量确保使用 Agg 后端

# 添加地震数据专用函数
def save_seismic_data(data, save_path):
    """保存地震数据为MAT文件"""
    sio.savemat(save_path, {'sr': data})


def calculate_psnr_float(gt, pred, data_range=None):
    """计算浮点数据的PSNR"""
    mse = np.mean((gt - pred) ** 2)
    if data_range is None:
        data_range = gt.max() - gt.min()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def calculate_ssim_float(gt, pred, data_range=None):
    """计算浮点数据的SSIM"""
    from skimage.metrics import structural_similarity as ssim
    if data_range is None:
        data_range = gt.max() - gt.min()
    return ssim(gt, pred, data_range=data_range)


'''
# --------------------------------------------
# training code for seismic data processing
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                             net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # 地震数据不需要border设置
    # border = opt.get('scale', 1)  # 注释掉或删除

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train samples: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    # 确保模型配置适配单通道地震数据
    if 'network_G' in opt and 'in_nc' in opt['network_G']:
        opt['network_G']['in_nc'] = 1  # 单通道输入
        opt['network_G']['out_nc'] = 1  # 单通道输出

    # # 确保模型配置适配3通道地震数据
    # if 'network_G' in opt and 'in_nc' in opt['network_G']:
    #     opt['network_G']['in_nc'] = 3  # 单通道输入
    #     opt['network_G']['out_nc'] = 3  # 单通道输出

    # model = define_Model(opt)
    model = ModelPlain3(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())
    # 创建损失记录文件
    if opt['rank'] == 0:
        loss_log_path = os.path.join(opt['path']['log'], 'loss_history.log')
        with open(loss_log_path, 'w') as f:
            f.write("Epoch,Train_Loss,Test_Loss,PSNR,SSIM\n")
    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # 从计算出的轮数开始
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)
        # 记录每个epoch的训练损失
        epoch_train_loss = 0.0
        num_batches = 0

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # 记录当前批次的损失
            logs = model.current_log()
            current_loss = logs.get('G_loss', 0)  # 根据实际损失键名调整
            epoch_train_loss += current_loss
            num_batches += 1

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:4d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing and visualization
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0
                test_loss = 0.0  # 添加测试损失记录
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    # 获取数据路径
                    data_path = test_data['L_path'][0]
                    data_name = os.path.splitext(os.path.basename(data_path))[0]

                    # 创建保存目录
                    # data_dir = os.path.join(opt['path']['images'], data_name, f"epoch_{( (current_step) //1203):04d}")
                    data_dir = os.path.join(opt['path']['images'], data_name, f"epoch_{((current_step) // 2430):03d}")
                    os.makedirs(data_dir, exist_ok=True)

                    # 处理数据
                    model.feed_data(test_data)
                    model.test()

                    # 获取结果
                    visuals = model.current_visuals()

                    # 地震数据不需要转换为uint8
                    # 直接使用浮点张量
                    E_tensor = visuals['E'].detach().cpu().float()
                    H_tensor = visuals['H'].detach().cpu().float()
                    input_tensor = test_data['L'].detach().cpu().float()
                    # 转换为numpy数组
                    E_data = E_tensor.squeeze().numpy()  # 移除批次和通道维度
                    H_data = H_tensor.squeeze().numpy()
                    input_data = input_tensor.squeeze().numpy()

                    # 计算测试损失
                    test_loss += model.G_lossfn(E_tensor, H_tensor).item()

                    # -----------------------
                    # 保存为MAT文件
                    # -----------------------
                    save_mat_path = os.path.join(data_dir, f'{data_name}.mat')
                    save_seismic_data(E_data, save_mat_path)

                    # -----------------------
                    # 计算地震数据质量指标
                    # -----------------------
                    data_range = H_data.max() - H_data.min()
                    current_psnr = calculate_psnr_float(H_data, E_data, data_range)
                    current_ssim = calculate_ssim_float(H_data, E_data, data_range)

                    logger.info('{:->4d}--> {:>10s} | PSNR: {:<4.2f}dB | SSIM: {:<4.4f}'.format(
                        idx, os.path.basename(data_path), current_psnr, current_ssim))

                    avg_psnr += current_psnr
                    avg_ssim += current_ssim

                    # -----------------------
                    # 创建图片展示
                    # -----------------------
                    fig, axs = plt.subplots(1, 3, figsize=(6, 2))

                    # # # 设置统一的颜色范围
                    # vmin = min(E_data.min(), input_data.min(), H_data.min())
                    # vmax = max(E_data.max(), input_data.max(), H_data.max())
                    # vmin = -0.3
                    # vmax = 0.3

                    # # 绘制输入图像
                    # im0 = axs[0].imshow(input_data, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
                    # axs[0].set_title('Migrated')
                    #
                    #
                    # # 绘制输出图像
                    # im1 = axs[1].imshow(E_data, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
                    # axs[1].set_title(f'Predicted\nPSNR: {current_psnr:.2f} dB\nSSIM: {current_ssim:.4f}')
                    #
                    #
                    # # 绘制目标图像
                    # im2 = axs[2].imshow(H_data, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
                    # axs[2].set_title('Reflectivity')

                    # 绘制输入图像
                    im0 = axs[0].imshow(input_data, cmap=plt.cm.gray, vmin=-0.3, vmax=0.3)


                    # 绘制输出图像
                    im1 = axs[1].imshow(E_data, cmap=plt.cm.gray, vmin=-0.6, vmax=0.6)


                    # 绘制目标图像
                    im2 = axs[2].imshow(H_data, cmap=plt.cm.gray, vmin=-0.6, vmax=0.6)
                    # 移除所有子图的刻度线和标签
                    for ax in axs:
                        ax.set_xticks([])  # 移除x轴刻度
                        ax.set_yticks([])  # 移除y轴刻度
                        ax.set_xticklabels([])  # 移除x轴标签
                        ax.set_yticklabels([])  # 移除y轴标签

                    # 保存图片
                    plt.tight_layout()
                    fig.savefig(os.path.join(data_dir, f'{data_name}.png'),bbox_inches='tight')
                    plt.close(fig)

                # 计算平均训练损失
                avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0

                # 计算平均测试损失和指标
                avg_test_loss = test_loss / idx
                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                # 记录到损失历史文件
                with open(loss_log_path, 'a') as f:
                    f.write(f"{epoch},{avg_train_loss},{avg_test_loss},{avg_psnr},{avg_ssim}\n")
                # testing log
                logger.info('<epoch:{:4d}, iter:{:8,d}, Avg Train Loss: {:.6f}, Avg Test Loss: {:.6f}, Avg PSNR: {:<.2f}dB, Avg SSIM: {:<.4f}\n'.format(
                            epoch + 1, current_step, avg_train_loss, avg_test_loss, avg_psnr, avg_ssim))
                # # testing log
                # logger.info('<epoch:{:3d}, iter:{:8,d}, Avg PSNR: {:<.2f}dB, Avg SSIM: {:<.4f}\n'.format(
                #     epoch+1, current_step, avg_psnr, avg_ssim))


if __name__ == '__main__':
    main()
