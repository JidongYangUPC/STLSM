# # -*-coding:utf-8-*-
# import argparse
# import cv2
# import glob
# import numpy as np
# from collections import OrderedDict
# import os
# import torch
# import requests
# # from models3.network_swinir import SwinIR as net
# from models3.network_swinir import EnhancedTripleInputSwinIR as net
# from utils import utils_image as util
# from scipy.io import loadmat, savemat
# import matplotlib.pyplot as plt  # 添加matplotlib用于可视化
# from skimage.metrics import structural_similarity as ssim_func
#
# # 添加浮点PSNR/SSIM计算函数
# def calculate_psnr_float(gt, pred, data_range=None):
#     """计算浮点数据的PSNR"""
#     mse = np.mean((gt - pred) ** 2)
#     if data_range is None:
#         data_range = gt.max() - gt.min()
#     if mse == 0:
#         return float('inf')
#     return 20 * np.log10(data_range / np.sqrt(mse))
#
# def calculate_ssim_float(gt, pred, data_range=None):
#     """计算浮点数据的SSIM"""
#     if data_range is None:
#         data_range = gt.max() - gt.min()
#
#         # 处理不同维度的数据
#     if gt.ndim == 2:
#         return ssim_func(gt, pred, data_range=data_range)
#     elif gt.ndim == 3:
#         # 多通道数据单独计算每通道SSIM后取平均
#         ssim_values = []
#         for i in range(gt.shape[2]):
#             channel_ssim = ssim_func(
#                 gt[:, :, i],
#                 pred[:, :, i],
#                 data_range=data_range
#             )
#             ssim_values.append(channel_ssim)
#         return np.mean(ssim_values)
#     else:
#         raise ValueError("Unsupported data dimensions")
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
#                                                                          'gray_dn, color_dn, jpeg_car, color_jpeg_car')
#     parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
#     parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
#     parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
#     parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
#                                                                              'Just used to differentiate two different settings in Table 2 of the paper. '
#                                                                              'Images are NOT tested patch by patch.')
#     parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
#     parser.add_argument('--model_path', type=str,
#                         default='model_zoo/swinir/001_classicalSR_DF2K_s48w8_SwinIR-M_x2.pth')
#     parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
#
#     parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
#     parser.add_argument('--tile', type=int, default=None,
#                         help='Tile size, None for no tile during testing (testing as a whole)')
#     parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
#     parser.add_argument('--save_dir', type=str, default=None, help='output directory to save results')
#     parser.add_argument('--window_size', type=int, default=8, help='Window size for Swin Transformer')
#     # parser.add_argument('--input_range', type=float, nargs=2, default=[-0.3, 0.3],
#     #                     help='Input data range for visualization')
#     # parser.add_argument('--output_range', type=float, nargs=2, default=[-0.7, 0.7],
#     #                     help='Output data range for visualization')
#     args = parser.parse_args()
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # set up model
#     if os.path.exists(args.model_path):
#         print(f'loading model from {args.model_path}')
#     else:
#         os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
#         url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(
#             os.path.basename(args.model_path))
#         r = requests.get(url, allow_redirects=True)
#         print(f'downloading model {args.model_path}')
#         open(args.model_path, 'wb').write(r.content)
#
#     model = define_model(args)
#     model.eval()
#     model = model.to(device)
#
#     # setup folder and path
#     folder, save_dir, border, window_size = setup(args)
#
#     # 创建两个子文件夹：一个用于保存mat文件，一个用于保存图片
#     save_dir_mat = os.path.join(save_dir, 'result_mat')
#     save_dir_img = os.path.join(save_dir, 'result_img')
#     os.makedirs(save_dir_mat, exist_ok=True)
#     os.makedirs(save_dir_img, exist_ok=True)
#
#     test_results = OrderedDict()
#     test_results['psnr'] = []
#     test_results['ssim'] = []
#     test_results['psnr_y'] = []
#     test_results['ssim_y'] = []
#     test_results['psnrb'] = []
#     test_results['psnrb_y'] = []
#     psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0
#
#     # for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
#     # # read image
#     #     imgname, img_lq, img_gt = get_image_pair(args, path)
#     #
#     # # 保存原始输入数据（用于后续可视化）
#     # input_data = img_lq.copy()
#     #
#     # # ================================
#     # img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
#     # img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
#     #
#     # # inference
#     # with torch.no_grad():
#     # # pad input image to be a multiple of window_size
#     #     _, _, h_old, w_old = img_lq.size()
#     # h_pad = (h_old // window_size + 1) * window_size - h_old
#     # w_pad = (w_old // window_size + 1) * window_size - w_old
#     # img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
#     # img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
#     # output = test(img_lq, model, args, window_size)
#     # output = output[..., :h_old * args.scale, :w_old * args.scale]
#     #
#     # # 保存输出图像为mat文件
#     # output_np = output.data.squeeze().float().cpu().numpy()
#     #
#     # # 如果是三通道，转换为HWC格式
#     # if output_np.ndim == 3:
#     #     output_np = np.transpose(output_np, (1, 2, 0))
#     #
#     # # 保存为mat文件（不进行归一化）
#     # mat_save_path = os.path.join(save_dir_mat, f'{imgname}.mat')
#     # savemat(mat_save_path, {'sr': output_np})
#     # print(f'Saved mat file: {mat_save_path}')
#     #
#     # # 保存为图片（使用指定的可视化方式）
#     # img_save_path = os.path.join(save_dir_img, f'{imgname}.png')
#     #
#     # # 准备可视化
#     # fig, axs = plt.subplots(1, 3, figsize=(6, 2))
#     #
#     # # 输入图像（Migrated）
#     # im0 = axs[0].imshow(input_data.squeeze(), cmap=plt.cm.gray,
#     #                     vmin=-0.3, vmax=-0.3)
#     # # axs[0].set_title('Migrated')
#     #
#     # # 输出图像（Predicted）
#     # # 如果是三通道，取第一个通道（假设三通道相同）
#     # if output_np.ndim == 3:
#     #     output_single = output_np[:, :, 0]
#     # else:
#     #     output_single = output_np
#     #
#     # im1 = axs[1].imshow(output_single, cmap=plt.cm.gray,
#     #                     vmin=-0.3, vmax=-0.3)
#     #
#     # # 如果有GT，计算PSNR和SSIM
#     # if img_gt is not None:
#     #     # 将GT转换为与输出相同的格式
#     #     if img_gt.ndim == 3:
#     #         gt_single = img_gt
#     #     else:
#     #         gt_single = img_gt[:, :, np.newaxis]
#     #
#     #     if output_np.ndim == 3:
#     #         output_single = output_np
#     #     else:
#     #         output_single = output_np[:, :, np.newaxis]
#     #     # 计算动态数据范围
#     #     data_range = gt_single.max() - gt_single.min()
#     #
#     #     # 使用浮点计算方法
#     #     psnr = calculate_psnr_float(gt_single, output_single, data_range)
#     #     ssim = calculate_ssim_float(gt_single, output_single, data_range)
#     #     # # 计算PSNR和SSIM
#     #     # psnr = util.calculate_psnr(output_single * 255, gt_single * 255)
#     #     # ssim = util.calculate_ssim(output_single * 255, gt_single * 255)
#     #
#     #     # axs[1].set_title(f'Predicted\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}')
#     #
#     #     # GT图像（Reflectivity）
#     #     im2 = axs[2].imshow(gt_single, cmap=plt.cm.gray,
#     #                         vmin=-0.3, vmax=0.3)
#     #     # axs[2].set_title('Reflectivity')
#     #
#     #     # 记录评估指标
#     #     test_results['psnr'].append(psnr)
#     #     test_results['ssim'].append(ssim)
#     # else:
#     #     axs[1].set_title('Predicted')
#     #     # 如果没有GT，隐藏第三个子图
#     #     axs[2].axis('off')
#     #
#     # # 移除所有子图的刻度线和标签
#     # for ax in axs:
#     #     ax.set_xticks([])
#     #     ax.set_yticks([])
#     #     ax.set_xticklabels([])
#     #     ax.set_yticklabels([])
#     #
#     # # 保存图片
#     # plt.tight_layout()
#     # fig.savefig(img_save_path, bbox_inches='tight', dpi=300)
#     # plt.close(fig)
#     # # print(f'Saved image: {img_save_path}')
#     #
#     # # 控制台输出评估结果
#     # if img_gt is not None:
#     #     print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(idx, imgname, psnr, ssim))
#     # else:
#     #     print('Testing {:d} {:20s}'.format(idx, imgname))
#     #
#     # # 如果有GT，总结评估结果
#     # if img_gt is not None and test_results['psnr']:
#     #     ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
#     #     ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
#     #     print('\n{} \n-- Average PSNR/SSIM: {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
#     for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
#         # read image
#         imgname, img_lq, img_gt = get_image_pair(args, path)
#
#         # 保存原始输入数据（用于后续可视化）
#         input_data = img_lq.copy()
#
#         # ================================
#         img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
#         img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)
#
#         # inference
#         with torch.no_grad():
#             _, _, h_old, w_old = img_lq.size()
#             h_pad = (h_old // window_size + 1) * window_size - h_old
#             w_pad = (w_old // window_size + 1) * window_size - w_old
#             img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
#             img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
#             output = test(img_lq, model, args, window_size)
#             output = output[..., :h_old * args.scale, :w_old * args.scale]
#
#         # 保存输出图像为mat文件
#         output_np = output.data.squeeze().float().cpu().numpy()
#         if output_np.ndim == 3:
#             output_np = np.transpose(output_np, (1, 2, 0))
#
#         mat_save_path = os.path.join(save_dir_mat, f'{imgname}.mat')
#         savemat(mat_save_path, {'sr': output_np})
#         # print(f'Saved mat file: {mat_save_path}')
#
#         # 保存为图片
#         img_save_path = os.path.join(save_dir_img, f'{imgname}.png')
#         fig, axs = plt.subplots(1, 3, figsize=(6, 2))
#
#         # 输入图像
#         im0 = axs[0].imshow(input_data.squeeze(), cmap=plt.cm.gray, vmin=-0.3, vmax=0.3)
#
#         # 输出图像
#         if output_np.ndim == 3:
#             output_single = output_np[:, :, 0]
#         else:
#             output_single = output_np
#         im1 = axs[1].imshow(output_single, cmap=plt.cm.gray, vmin=-0.7, vmax=0.7)
#
#         # 如果有GT，计算PSNR和SSIM
#         if img_gt is not None:
#             if img_gt.ndim == 3:
#                 gt_single = img_gt
#             else:
#                 gt_single = img_gt[:, :, np.newaxis]
#
#             if output_np.ndim == 3:
#                 output_single = output_np
#             else:
#                 output_single = output_np[:, :, np.newaxis]
#
#             data_range = gt_single.max() - gt_single.min()
#             psnr = calculate_psnr_float(gt_single, output_single, data_range)
#             ssim = calculate_ssim_float(gt_single, output_single, data_range)
#
#             im2 = axs[2].imshow(gt_single, cmap=plt.cm.gray, vmin=-0.7, vmax=0.7)
#             test_results['psnr'].append(psnr)
#             test_results['ssim'].append(ssim)
#         else:
#             axs[1].set_title('Predicted')
#             axs[2].axis('off')
#
#         # 移除刻度
#         for ax in axs:
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#
#         # 保存图片
#         plt.tight_layout()
#         fig.savefig(img_save_path, bbox_inches='tight')
#         plt.close(fig)
#
#         # 输出当前结果
#         if img_gt is not None:
#             print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(idx+1, imgname, psnr, ssim))
#         else:
#             print('Testing {:d} {:20s}'.format(idx+1, imgname))
#
#     # 循环结束后计算平均值
#     if img_gt is not None and test_results['psnr']:
#         ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
#         ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
#         print('\n{} \n-- Average PSNR/SSIM: {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
#
# def define_model(args):
#     # 001 classical image sr
#     if args.task == 'classical_sr':
#         model = net(upscale=args.scale, in_chans=1, img_size=args.training_patch_size, window_size=args.window_size,
#                     img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6, 6, 6],
#                     mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
#         param_key_g = 'params'
#
#     # 002 lightweight image sr
#     # use 'pixelshuffledirect' to save parameters
#     elif args.task == 'lightweight_sr':
#         model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
#                 img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
#                 mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
#         param_key_g = 'params'
#
#     # 003 real-world image sr
#     elif args.task == 'real_sr':
#         if not args.large_model:
#             # use 'nearest+conv' to avoid block artifacts
#             model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
#                         img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
#                         mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
#         else:
#             # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
#             model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
#                         img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
#                         num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
#                         mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
#         param_key_g = 'params_ema'
#
#     # 004 grayscale image denoising
#     elif args.task == 'gray_dn':
#         model = net(upscale=1, in_chans=1, img_size=256, window_size=4,
#                     img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=120, num_heads=[6, 6, 6, 6, 6, 6],
#                     mlp_ratio=2, upsampler='', resi_connection='1conv')
#         param_key_g = 'params'
#
#     # 005 color image denoising
#     elif args.task == 'color_dn':
#         model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
#                     img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
#                     mlp_ratio=2, upsampler='', resi_connection='1conv')
#         param_key_g = 'params'
#
#     # 006 JPEG compression artifact reduction
#     # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
#     elif args.task == 'jpeg_car':
#         model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
#                     img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
#                     mlp_ratio=2, upsampler='', resi_connection='1conv')
#         param_key_g = 'params'
#
#     pretrained_model = torch.load(args.model_path)
#     model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
#                           strict=True)
#
#     return model
#
#
# def setup(args):
#     # 根据任务类型初始化默认的folder, border, window_size
#     if args.task in ['classical_sr', 'lightweight_sr']:
#         folder = args.folder_gt
#         save_dir = f'results/swinir_{args.task}_x{args.scale}'
#         border = args.scale
#         window_size = 8
#     # 其他任务设置保持不变...
#     # 为简洁起见，这里省略其他任务设置
#     # 003 real-world image sr
#     elif args.task in ['real_sr']:
#         save_dir = f'results/swinir_{args.task}_x{args.scale}'
#         if args.large_model:
#             save_dir += '_large'
#         folder = args.folder_lq
#         border = 0
#         window_size = 8
#
#     # 004 grayscale image denoising/ 005 color image denoising
#     elif args.task in ['gray_dn', 'color_dn']:
#         save_dir = f'results/swinir_{args.task}_noise{args.noise}'
#         folder = args.folder_gt
#         border = 0
#         window_size = 8
#
#     # 006 JPEG compression artifact reduction
#     elif args.task in ['jpeg_car']:
#         save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}'
#         folder = args.folder_gt
#         border = 0
#         window_size = 7
#     # 如果用户指定了 --save_dir，覆盖save_dir的值
#     if args.save_dir:
#         save_dir = args.save_dir
#
#     # 确保目录存在
#     os.makedirs(save_dir, exist_ok=True)
#     return folder, save_dir, border, window_size
#
#
# # def get_image_pair(args, path):
# #     (imgname, imgext) = os.path.splitext(os.path.basename(path))
# #     # 处理classical_sr任务 - 从mat文件加载
# #     if args.task in ['classical_sr', 'lightweight_sr']:
# #         # 加载低分辨率输入
# #         lr_path = path
# #         mat_data_lq = loadmat(lr_path)
# #
# #         # 获取数据 - 假设数据存储在'data'键中
# #         img_lq = mat_data_lq['im_lq']
# #         # # 确保数据是二维或三维数组
# #         # if img_lq.ndim not in [2, 3]:
# #         #     raise ValueError(f"Invalid data dimensions in {lr_path}: {img_lq.shape}")
# #
# #         # # 如果是二维数组，添加通道维度
# #         # if img_lq.ndim == 2:
# #         #     img_lq = img_lq[:, :, np.newaxis]
# #
# #         # 尝试加载对应的GT（如果有）
# #         img_gt = None
# #         if args.folder_gt:
# #             gt_path = os.path.join(args.folder_gt, f"{imgname}{imgext}")
# #             if os.path.exists(gt_path):
# #                 mat_data_gt = loadmat(gt_path)
# #                 img_gt = mat_data_gt['im_gt']
# #
# #
# #                 # # 确保数据是二维或三维数组
# #                 # if img_gt.ndim not in [2, 3]:
# #                 #     raise ValueError(f"Invalid GT data dimensions in {gt_path}: {img_gt.shape}")
# #                 #
# #                 # # 如果是二维数组，添加通道维度
# #                 # if img_gt.ndim == 2:
# #                 #     img_gt = img_gt[:, :, np.newaxis]
# #
# #         return imgname, img_lq, img_gt
#
# def get_image_pair(args, path):
#     (imgname, imgext) = os.path.splitext(os.path.basename(path))
#     # print(f"Processing: {imgname}")
#
#     # 处理classical_sr任务 - 从mat文件加载
#     if args.task in ['classical_sr', 'lightweight_sr']:
#         # 加载高分辨率图像 (GT)
#         mat_data_gt = loadmat(path)
#         img_gt = mat_data_gt['im_gt']
#         # print(f"Loaded GT data: shape={img_gt.shape}, dtype={img_gt.dtype}")
#
#         # 确保数据是浮点类型
#         img_gt = img_gt.astype(np.float32)
#
#         # 根据高分辨率文件名推导低分辨率文件名
#         # 原始命名规则: LR文件名 = GT文件名 + "x{scale}" + 扩展名
#         lr_filename = f"{imgname}x{args.scale}{imgext}"
#         lr_path = os.path.join(args.folder_lq, lr_filename)
#         # print(f"Looking for LR at: {lr_path}")
#
#         # 加载低分辨率图像 (LQ)
#         if os.path.exists(lr_path):
#             mat_data_lq = loadmat(lr_path)
#             img_lq = mat_data_lq['im_lq']
#             # print(f"Loaded LR data: shape={img_lq.shape}, dtype={img_lq.dtype}")
#
#             # 确保数据是浮点类型
#             img_lq = img_lq.astype(np.float32)
#         else:
#             raise FileNotFoundError(f"LR file not found: {lr_path}")
#
#         # 确保数据有正确的维度
#         if img_gt.ndim == 2:
#             img_gt = img_gt[:, :, np.newaxis]
#         elif img_gt.ndim == 3 and img_gt.shape[2] > 3:
#             img_gt = img_gt[:, :, :3] if img_gt.shape[2] >= 3 else img_gt[:, :, 0:1]
#
#         if img_lq.ndim == 2:
#             img_lq = img_lq[:, :, np.newaxis]
#         elif img_lq.ndim == 3 and img_lq.shape[2] > 3:
#             img_lq = img_lq[:, :, :3] if img_lq.shape[2] >= 3 else img_lq[:, :, 0:1]
#
#         return imgname, img_lq, img_gt
#     # 其他任务的数据加载保持不变...
#     # 为简洁起见，这里省略其他任务的数据加载
#
# # 003 real-world image sr (load lq image only)
#     elif args.task in ['real_sr']:
#         img_gt = None
#         img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
#
#     # 004 grayscale image denoising (load gt image and generate lq image on-the-fly)
#     elif args.task in ['gray_dn']:
#         img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
#         np.random.seed(seed=0)
#         img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
#         img_gt = np.expand_dims(img_gt, axis=2)
#         img_lq = np.expand_dims(img_lq, axis=2)
#
#     # 005 color image denoising (load gt image and generate lq image on-the-fly)
#     elif args.task in ['color_dn']:
#         img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
#         np.random.seed(seed=0)
#         img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
#
#     # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
#     elif args.task in ['jpeg_car']:
#         img_gt = cv2.imread(path, 0)
#         result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
#         img_lq = cv2.imdecode(encimg, 0)
#         img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
#         img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.
#
#     return imgname, img_lq, img_gt
#
# def test(img_lq, model, args, window_size):
#     if args.tile is None:
#         # test the image as a whole
#         output = model(img_lq)
#     else:
#         # test the image tile by tile
#         b, c, h, w = img_lq.size()
#         tile = min(args.tile, h, w)
#         assert tile % window_size == 0, "tile size should be a multiple of window_size"
#         tile_overlap = args.tile_overlap
#         sf = args.scale
#
#         stride = tile - tile_overlap
#         h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
#         w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
#         E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
#         W = torch.zeros_like(E)
#
#         for h_idx in h_idx_list:
#             for w_idx in w_idx_list:
#                 in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
#                 out_patch = model(in_patch)
#                 out_patch_mask = torch.ones_like(out_patch)
#
#                 E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
#                 W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
#         output = E.div_(W)
#
#     return output
#
#
# if __name__ == '__main__':
#     main()




# -*-coding:utf-8-*-
import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
# from models3.network_swinir import SwinIR as net
from models3.network_swinir import EnhancedTripleInputSwinIR as net
from utils import utils_image as util
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt  # 添加matplotlib用于可视化
from skimage.metrics import structural_similarity as ssim_func

# 添加浮点PSNR/SSIM计算函数
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
    if data_range is None:
        data_range = gt.max() - gt.min()

        # 处理不同维度的数据
    if gt.ndim == 2:
        return ssim_func(gt, pred, data_range=data_range)
    elif gt.ndim == 3:
        # 多通道数据单独计算每通道SSIM后取平均
        ssim_values = []
        for i in range(gt.shape[2]):
            channel_ssim = ssim_func(
                gt[:, :, i],
                pred[:, :, i],
                data_range=data_range
            )
            ssim_values.append(channel_ssim)
        return np.mean(ssim_values)
    else:
        raise ValueError("Unsupported data dimensions")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                         'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                                                             'Just used to differentiate two different settings in Table 2 of the paper. '
                                                                             'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DF2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_illumination', type=str, default=None, help='input illumination test image folder')
    parser.add_argument('--folder_velocity', type=str, default=None, help='input velocity test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--save_dir', type=str, default=None, help='output directory to save results')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for Swin Transformer')
    # parser.add_argument('--input_range', type=float, nargs=2, default=[-0.3, 0.3],
    #                     help='Input data range for visualization')
    # parser.add_argument('--output_range', type=float, nargs=2, default=[-0.7, 0.7],
    #                     help='Output data range for visualization')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(
            os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)

    # 创建两个子文件夹：一个用于保存mat文件，一个用于保存图片
    save_dir_mat = os.path.join(save_dir, 'result_mat')
    save_dir_img = os.path.join(save_dir, 'result_img')
    os.makedirs(save_dir_mat, exist_ok=True)
    os.makedirs(save_dir_img, exist_ok=True)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []
    psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0


    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_illumination, img_velocity,img_gt = get_image_pair(args, path)

        # 保存原始输入数据（用于后续可视化）
        input_data = img_lq.copy()

        # ================================
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        img_illumination = np.transpose(img_illumination if img_illumination.shape[2] == 1 else img_illumination[:, :, [2, 1, 0]], (2, 0, 1))
        img_illumination = torch.from_numpy(img_illumination).float().unsqueeze(0).to(device)

        img_velocity = np.transpose(img_velocity if img_velocity.shape[2] == 1 else img_velocity[:, :, [2, 1, 0]],(2, 0, 1))
        img_velocity = torch.from_numpy(img_velocity).float().unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            # _, _, h_old, w_old = img_lq.size()
            # h_pad = (h_old // window_size + 1) * window_size - h_old
            # w_pad = (w_old // window_size + 1) * window_size - w_old
            # img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            # img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            # output = test(img_lq, model, args, window_size)
            # output = output[..., :h_old * args.scale, :w_old * args.scale]

            # 对每个输入分别进行padding
            inputs = [img_lq, img_illumination, img_velocity]
            padded_inputs = []

            for inp in inputs:
                _, _, h_old, w_old = inp.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                inp_padded = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
                inp_padded = torch.cat([inp_padded, torch.flip(inp_padded, [3])], 3)[:, :, :, :w_old + w_pad]
                padded_inputs.append(inp_padded)

            # 使用三个输入进行推理
            output = test(padded_inputs, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # 保存输出图像为mat文件
        output_np = output.data.squeeze().float().cpu().numpy()
        if output_np.ndim == 3:
            output_np = np.transpose(output_np, (1, 2, 0))

        mat_save_path = os.path.join(save_dir_mat, f'{imgname}.mat')
        savemat(mat_save_path, {'sr': output_np})
        # print(f'Saved mat file: {mat_save_path}')

        # 保存为图片
        img_save_path = os.path.join(save_dir_img, f'{imgname}.png')
        fig, axs = plt.subplots(1, 3, figsize=(6, 2))

        # 输入图像
        im0 = axs[0].imshow(input_data.squeeze(), cmap=plt.cm.gray, vmin=-0.3, vmax=0.3)

        # 输出图像
        if output_np.ndim == 3:
            output_single = output_np[:, :, 0]
        else:
            output_single = output_np
        im1 = axs[1].imshow(output_single, cmap=plt.cm.gray, vmin=-0.3, vmax=0.3)

        # 如果有GT，计算PSNR和SSIM
        if img_gt is not None:
            if img_gt.ndim == 3:
                gt_single = img_gt
            else:
                gt_single = img_gt[:, :, np.newaxis]

            if output_np.ndim == 3:
                output_single = output_np
            else:
                output_single = output_np[:, :, np.newaxis]

            data_range = gt_single.max() - gt_single.min()
            psnr = calculate_psnr_float(gt_single, output_single, data_range)
            ssim = calculate_ssim_float(gt_single, output_single, data_range)

            im2 = axs[2].imshow(gt_single, cmap=plt.cm.gray, vmin=-0.3, vmax=0.3)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
        else:
            axs[1].set_title('Predicted')
            axs[2].axis('off')

        # 移除刻度
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # 保存图片
        plt.tight_layout()
        fig.savefig(img_save_path, bbox_inches='tight')
        plt.close(fig)

        # 输出当前结果
        if img_gt is not None:
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(idx+1, imgname, psnr, ssim))
        else:
            print('Testing {:d} {:20s}'.format(idx+1, imgname))

    # 循环结束后计算平均值
    if img_gt is not None and test_results['psnr']:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM: {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))

def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=1, img_size=args.training_patch_size, window_size=args.window_size,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=256, window_size=4,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=120, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)

    return model


def setup(args):
    # 根据任务类型初始化默认的folder, border, window_size
    if args.task in ['classical_sr', 'lightweight_sr']:
        folder = args.folder_gt
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        border = args.scale
        window_size = 8
        # 检查必要的文件夹参数
        if not args.folder_lq:
            raise ValueError("--folder_lq is required for classical_sr and lightweight_sr tasks")
        if not args.folder_illumination:
            raise ValueError("--folder_illumination is required for triple input model")
        if not args.folder_velocity:
            raise ValueError("--folder_velocity is required for triple input model")

    # 其他任务设置保持不变...
    # 为简洁起见，这里省略其他任务设置
    # 003 real-world image sr
    elif args.task in ['real_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8

    # 004 grayscale image denoising/ 005 color image denoising
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'results/swinir_{args.task}_noise{args.noise}'
        folder = args.folder_gt
        border = 0
        window_size = 8

    # 006 JPEG compression artifact reduction
    elif args.task in ['jpeg_car']:
        save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}'
        folder = args.folder_gt
        border = 0
        window_size = 7
    # 如果用户指定了 --save_dir，覆盖save_dir的值
    if args.save_dir:
        save_dir = args.save_dir

    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    return folder, save_dir, border, window_size


# def get_image_pair(args, path):
#     (imgname, imgext) = os.path.splitext(os.path.basename(path))
#     # 处理classical_sr任务 - 从mat文件加载
#     if args.task in ['classical_sr', 'lightweight_sr']:
#         # 加载低分辨率输入
#         lr_path = path
#         mat_data_lq = loadmat(lr_path)
#
#         # 获取数据 - 假设数据存储在'data'键中
#         img_lq = mat_data_lq['im_lq']
#         # # 确保数据是二维或三维数组
#         # if img_lq.ndim not in [2, 3]:
#         #     raise ValueError(f"Invalid data dimensions in {lr_path}: {img_lq.shape}")
#
#         # # 如果是二维数组，添加通道维度
#         # if img_lq.ndim == 2:
#         #     img_lq = img_lq[:, :, np.newaxis]
#
#         # 尝试加载对应的GT（如果有）
#         img_gt = None
#         if args.folder_gt:
#             gt_path = os.path.join(args.folder_gt, f"{imgname}{imgext}")
#             if os.path.exists(gt_path):
#                 mat_data_gt = loadmat(gt_path)
#                 img_gt = mat_data_gt['im_gt']
#
#
#                 # # 确保数据是二维或三维数组
#                 # if img_gt.ndim not in [2, 3]:
#                 #     raise ValueError(f"Invalid GT data dimensions in {gt_path}: {img_gt.shape}")
#                 #
#                 # # 如果是二维数组，添加通道维度
#                 # if img_gt.ndim == 2:
#                 #     img_gt = img_gt[:, :, np.newaxis]
#
#         return imgname, img_lq, img_gt

def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    # print(f"Processing: {imgname}")

    # 处理classical_sr任务 - 从mat文件加载
    if args.task in ['classical_sr', 'lightweight_sr']:
        # 加载高分辨率图像 (GT)
        mat_data_gt = loadmat(path)
        img_gt = mat_data_gt['im_gt']
        # print(f"Loaded GT data: shape={img_gt.shape}, dtype={img_gt.dtype}")

        # 确保数据是浮点类型
        img_gt = img_gt.astype(np.float32)

        # 根据高分辨率文件名推导低分辨率文件名
        # 原始命名规则: LR文件名 = GT文件名 + "x{scale}" + 扩展名
        lr_filename = f"{imgname}x{args.scale}{imgext}"
        lr_path = os.path.join(args.folder_lq, lr_filename)
        # print(f"Looking for LR at: {lr_path}")

        # 加载低分辨率图像 (LQ)
        if os.path.exists(lr_path):
            mat_data_lq = loadmat(lr_path)
            img_lq = mat_data_lq['im_lq']
            # print(f"Loaded LR data: shape={img_lq.shape}, dtype={img_lq.dtype}")

            # 确保数据是浮点类型
            img_lq = img_lq.astype(np.float32)
        else:
            raise FileNotFoundError(f"LR file not found: {lr_path}")
        # 加载震源照明数据 - 使用与LQ文件相同的文件名
        illumination_filename = lr_filename
        illumination_path = os.path.join(args.folder_illumination, illumination_filename)
        if os.path.exists(illumination_path):
            mat_data_illumination = loadmat(illumination_path)
            img_illumination = mat_data_illumination['illumination']
            img_illumination = img_illumination.astype(np.float32)
        else:
            raise FileNotFoundError(f"Illumination file not found: {illumination_path}")

        # 加载偏移速度模型
        velocity_filename = lr_filename
        velocity_path = os.path.join(args.folder_velocity, velocity_filename)
        if os.path.exists(velocity_path):
            mat_data_velocity = loadmat(velocity_path)
            img_velocity = mat_data_velocity['velocity']
            img_velocity = img_velocity.astype(np.float32)
        else:
            raise FileNotFoundError(f"Velocity file not found: {velocity_path}")



        # 确保数据有正确的维度
        if img_gt.ndim == 2:
            img_gt = img_gt[:, :, np.newaxis]
        elif img_gt.ndim == 3 and img_gt.shape[2] > 3:
            img_gt = img_gt[:, :, :3] if img_gt.shape[2] >= 3 else img_gt[:, :, 0:1]

        if img_lq.ndim == 2:
            img_lq = img_lq[:, :, np.newaxis]
        elif img_lq.ndim == 3 and img_lq.shape[2] > 3:
            img_lq = img_lq[:, :, :3] if img_lq.shape[2] >= 3 else img_lq[:, :, 0:1]

        if img_illumination.ndim == 2:
            img_illumination = img_illumination[:, :, np.newaxis]
        elif img_illumination.ndim == 3 and img_illumination.shape[2] > 3:
            img_illumination = img_illumination[:, :, :3] if img_illumination.shape[2] >= 3 else img_illumination[:, :,
                                                                                                 0:1]

        if img_velocity.ndim == 2:
            img_velocity = img_velocity[:, :, np.newaxis]
        elif img_velocity.ndim == 3 and img_velocity.shape[2] > 3:
            img_velocity = img_velocity[:, :, :3] if img_velocity.shape[2] >= 3 else img_velocity[:, :, 0:1]

        return imgname, img_lq, img_illumination, img_velocity,img_gt
    # 其他任务的数据加载保持不变...
    # 为简洁起见，这里省略其他任务的数据加载

# 003 real-world image sr (load lq image only)
    elif args.task in ['real_sr']:
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # 004 grayscale image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ['gray_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

    # 005 color image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ['color_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)

    # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ['jpeg_car']:
        img_gt = cv2.imread(path, 0)
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

    return imgname, img_lq, img_gt

def test(inputs, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(*inputs)
    else:
        # test the image tile by tile
        b, c, h, w = inputs[0].size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(inputs[0])
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                # 对每个输入提取patch
                in_patches = []
                for inp in inputs:
                    in_patch = inp[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    in_patches.append(in_patch)

                # 使用三个输入进行推理
                out_patch = model(*in_patches)
                out_patch_mask = torch.ones_like(out_patch)
                # in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                # out_patch = model(in_patch)
                # out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


if __name__ == '__main__':
    main()