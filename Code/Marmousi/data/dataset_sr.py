# import random
# import numpy as np
# import scipy.io as sio
# import torch.utils.data as data
# import utils.utils_image as util
# import os
#
# class DatasetSR(data.Dataset):
#     '''
#     # -----------------------------------------
#     # Get L/H for SISR.
#     # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
#     # -----------------------------------------
#     # e.g., SRResNet
#     # -----------------------------------------
#     '''
#
#     def __init__(self, opt):
#         super(DatasetSR, self).__init__()
#         self.opt = opt
#         self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
#         self.sf = opt['scale'] if opt['scale'] else 4
#         self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
#         self.L_size = self.patch_size // self.sf
#
#         # ------------------------------------
#         # get paths of L/H
#         # ------------------------------------
#         self.paths_H = util.get_image_paths(opt['dataroot_H'])
#         self.paths_L = util.get_image_paths(opt['dataroot_L'])
#
#         assert self.paths_H, 'Error: H path is empty.'
#         if self.paths_L and self.paths_H:
#             assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))
#
#     def __getitem__(self, index):
#
#         L_path = None
#         # ------------------------------------
#         # get H image
#         # ------------------------------------
#         H_path = self.paths_H[index]
#         img_H = util.imread_uint(H_path, self.n_channels)
#         img_H = util.uint2single(img_H)
#
#         # ------------------------------------
#         # modcrop
#         # ------------------------------------
#         img_H = util.modcrop(img_H, self.sf)
#
#         # ------------------------------------
#         # get L image
#         # ------------------------------------
#         if self.paths_L:
#             # --------------------------------
#             # directly load L image
#             # --------------------------------
#             L_path = self.paths_L[index]
#             img_L = util.imread_uint(L_path, self.n_channels)
#             img_L = util.uint2single(img_L)
#
#         else:
#             # --------------------------------
#             # sythesize L image via matlab's bicubic
#             # --------------------------------
#             H, W = img_H.shape[:2]
#             img_L = util.imresize_np(img_H, 1 / self.sf, True)
#
#         # ------------------------------------
#         # if train, get L/H patch pair
#         # ------------------------------------
#         if self.opt['phase'] == 'train':
#
#             H, W, C = img_L.shape
#
#             # --------------------------------
#             # randomly crop the L patch
#             # --------------------------------
#             rnd_h = random.randint(0, max(0, H - self.L_size))
#             rnd_w = random.randint(0, max(0, W - self.L_size))
#             img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
#
#             # --------------------------------
#             # crop corresponding H patch
#             # --------------------------------
#             rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
#             img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
#
#             # --------------------------------
#             # augmentation - flip and/or rotate
#             # --------------------------------
#             mode = random.randint(0, 7)
#             img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)
#
#         # ------------------------------------
#         # L/H pairs, HWC to CHW, numpy to tensor
#         # ------------------------------------
#         img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
#
#         if L_path is None:
#             L_path = H_path
#
#         return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
#
#     def __len__(self):
#         return len(self.paths_H)


#three_information
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import scipy.io as sio
import os


class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for seismic data processing.
    # Modified to handle .mat files with seismic data
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf
        # self.n_channels = 1  # Seismic data is single-channel
        # self.patch_size = opt.get('H_size', 256)  # Default patch size for seismic data
        # 定义不同数据类型的键名
        self.key_H = opt.get('key_H', 'im_gt')  # HR数据的键名
        self.key_L = opt.get('key_L', 'im_lq')  # LR数据的键名
        self.key_illumination = opt.get('key_illumination', 'illumination')  # 震源照明数据的键名
        self.key_velocity = opt.get('key_velocity', 'velocity')  # 偏移速度模型数据的键名
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = self.get_mat_paths(opt['dataroot_H'])
        self.paths_L = self.get_mat_paths(opt['dataroot_L']) if opt.get('dataroot_L') else None
        self.paths_illumination = self.get_mat_paths(opt['dataroot_illumination']) if opt.get(
            'dataroot_illumination') else None
        self.paths_velocity = self.get_mat_paths(opt['dataroot_velocity']) if opt.get('dataroot_velocity') else None

        # assert self.paths_H, 'Error: H path is empty.'
        # if self.paths_L and self.paths_H:
        #     assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L),
        #                                                                                    len(self.paths_H))
        # 验证数据路径
        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L),
                                                                                           len(self.paths_H))
        if self.paths_illumination:
            assert len(self.paths_illumination) == len(self.paths_H), 'Illumination/H mismatch - {}, {}.'.format(
                len(self.paths_illumination), len(self.paths_H))
        if self.paths_velocity:
            assert len(self.paths_velocity) == len(self.paths_H), 'Velocity/H mismatch - {}, {}.'.format(
                len(self.paths_velocity), len(self.paths_H))

        # For seismic data, we don't use scale factor in the same way as images
        self.scale = opt.get('scale', 1)  # Keep for compatibility, but not used for cropping

    def get_mat_paths(self, dataroot):
        """Get paths to all .mat files in the directory"""
        paths = []
        if dataroot and os.path.exists(dataroot):
            for fname in sorted(os.listdir(dataroot)):
                if fname.endswith('.mat'):
                    paths.append(os.path.join(dataroot, fname))
        return paths

    def load_seismic_data(self, path, key):
        """Load seismic data from .mat file"""
        try:
            data = sio.loadmat(path)
            # 尝试使用指定的键名获取数据
            if key in data:
                seismic = data[key].astype(np.float32)
            else:
                # 如果指定的键名不存在，尝试使用第一个非内部变量
                for k in data.keys():
                    if not k.startswith('__'):
                        seismic = data[k].astype(np.float32)
                        break
                else:
                    raise ValueError(f"No valid data found in {path}")

            # 确保数据是2D或3D的
            if seismic.ndim == 2:
                seismic = seismic[..., np.newaxis]  # 添加通道维度: (H, W) -> (H, W, 1)
            elif seismic.ndim == 3 and seismic.shape[2] > 3:
                # 如果通道数大于3，取前3个通道
                seismic = seismic[:, :, :3]

            return seismic
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise

    def __getitem__(self, index):
        # ------------------------------------
        # get H (target) seismic data
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = self.load_seismic_data(H_path, self.key_H)

        # ------------------------------------
        # get L (input) seismic data
        # ------------------------------------
        if self.paths_L:
            L_path = self.paths_L[index]
            img_L = self.load_seismic_data(L_path, self.key_L)
        else:
            # If no L path provided, use H as L (for self-supervised learning)
            img_L = img_H.copy()
            L_path = H_path

        # ------------------------------------
        # get illumination data
        # ------------------------------------
        if self.paths_illumination:
            illumination_path = self.paths_illumination[index]
            img_illumination = self.load_seismic_data(illumination_path, self.key_illumination)
        else:
            # If no illumination path provided, create a dummy array
            img_illumination = np.ones_like(img_L)
            illumination_path = "dummy_illumination"

        # ------------------------------------
        # get velocity data
        # ------------------------------------
        if self.paths_velocity:
            velocity_path = self.paths_velocity[index]
            img_velocity = self.load_seismic_data(velocity_path, self.key_velocity)
        else:
            # If no velocity path provided, create a dummy array
            img_velocity = np.ones_like(img_L)
            velocity_path = "dummy_velocity"

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            H, W, _ = img_L.shape
            min_H = min(img_L.shape[0], img_illumination.shape[0], img_velocity.shape[0], img_H.shape[0])
            min_W = min(img_L.shape[1], img_illumination.shape[1], img_velocity.shape[1], img_H.shape[1])

            # 如果尺寸不匹配，调整到最小尺寸
            if min_H < H or min_W < W:
                H, W = min_H, min_W
                img_L = img_L[:H, :W, :]
                img_illumination = img_illumination[:H, :W, :]
                img_velocity = img_velocity[:H, :W, :]
                img_H = img_H[:H, :W, :]

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))

            img_L = img_L[rnd_h:rnd_h + self.patch_size,
                        rnd_w:rnd_w + self.patch_size, :]
            img_illumination = img_illumination[rnd_h:rnd_h + self.patch_size,
                               rnd_w:rnd_w + self.patch_size, :]

            img_velocity = img_velocity[rnd_h:rnd_h + self.patch_size,
                           rnd_w:rnd_w + self.patch_size, :]

            img_H = img_H[rnd_h:rnd_h + self.patch_size,
                        rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            if self.opt.get('use_flip', True):
                mode = random.randint(0, 7)
                img_L = util.augment_seismic(img_L, mode=mode)
                img_illumination = util.augment_seismic(img_illumination, mode=mode)
                img_velocity = util.augment_seismic(img_velocity, mode=mode)
                img_H = util.augment_seismic(img_H, mode=mode)

            # --------------------------------
            # add noise if specified
            # --------------------------------
            if self.opt.get('use_noise', False):
                noise_level = self.opt.get('noise_level', 0.01)
                noise = np.random.normal(0, noise_level, img_L.shape)
                img_L = img_L + noise

        # ------------------------------------
        # Convert to PyTorch tensor
        # ------------------------------------
        # HWC to CHW: (H, W, C) -> (C, H, W)
        img_H = torch.from_numpy(img_H.transpose(2, 0, 1).copy())
        img_L = torch.from_numpy(img_L.transpose(2, 0, 1).copy())
        img_illumination = torch.from_numpy(img_illumination.transpose(2, 0, 1).copy())
        img_velocity = torch.from_numpy(img_velocity.transpose(2, 0, 1).copy())

        return {'L': img_L,
                'illumination': img_illumination,
                'velocity': img_velocity,
                'H': img_H,
                'L_path': L_path,
                'illumination_path': illumination_path,
                'velocity_path': velocity_path,
                'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)



# # # shuju_three_no_information
# import random
# import numpy as np
# import torch
# import torch.utils.data as data
# import utils.utils_image as util
# import scipy.io as sio
# import os
#
#
# class DatasetSR(data.Dataset):
#     '''
#     # -----------------------------------------
#     # Get L/H for seismic data processing.
#     # Modified to handle .mat files with seismic data
#     # -----------------------------------------
#     '''
#
#     def __init__(self, opt):
#         super(DatasetSR, self).__init__()
#         self.opt = opt
#         self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
#         self.sf = opt['scale'] if opt['scale'] else 4
#         self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
#         self.L_size = self.patch_size // self.sf
#         # self.n_channels = 1  # Seismic data is single-channel
#         # self.patch_size = opt.get('H_size', 256)  # Default patch size for seismic data
#         # 定义不同数据类型的键名
#         self.key_H = opt.get('key_H', 'im_gt')  # HR数据的键名
#         self.key_L = opt.get('key_L', 'im_lq')  # LR数据的键名
#         self.key_illumination = opt.get('key_illumination', 'illumination')  # 震源照明数据的键名
#         self.key_velocity = opt.get('key_velocity', 'velocity')  # 偏移速度模型数据的键名
#         # ------------------------------------
#         # get paths of L/H
#         # ------------------------------------
#         # 获取LR后缀（注意：是小写的'x2'）
#         self.lr_suffix = opt.get('lr_suffix', 'x2')  # 修改：小写x
#         # 获取所有数据路径
#         # ------------------------------------
#         # 获取HR数据路径
#         self.paths_H = self.get_mat_paths(opt['dataroot_H'])
#         assert self.paths_H, 'Error: H path is empty.'
#
#         # 创建文件名到路径的映射字典（不包括扩展名）
#         self.h_name_to_path = {}
#         self.h_name_to_idx = {}
#         for idx, path in enumerate(self.paths_H):
#             base_name = os.path.splitext(os.path.basename(path))[0]  # 例如: "example1"
#             self.h_name_to_path[base_name] = path
#             self.h_name_to_idx[base_name] = idx
#
#         # 获取LR数据路径并创建映射
#         if opt.get('dataroot_L'):
#             self.paths_L_all = self.get_mat_paths(opt['dataroot_L'])
#             # 创建LR文件名映射
#             self.l_name_to_path = {}
#             for path in self.paths_L_all:
#                 base_name = os.path.splitext(os.path.basename(path))[0]  # 例如: "example1X2"
#                 self.l_name_to_path[base_name] = path
#             print(f"LR数据加载: {len(self.paths_L_all)}个文件")
#         else:
#             self.paths_L_all = None
#             self.l_name_to_path = {}
#             print("警告: 未提供LR数据路径")
#
#         # 获取Illumination数据路径并创建映射
#         if opt.get('dataroot_illumination'):
#             self.paths_illumination_all = self.get_mat_paths(opt['dataroot_illumination'])
#             self.illumination_name_to_path = {}
#             for path in self.paths_illumination_all:
#                 base_name = os.path.splitext(os.path.basename(path))[0]  # 例如: "example1"
#                 self.illumination_name_to_path[base_name] = path
#             print(f"Illumination数据加载: {len(self.paths_illumination_all)}个文件")
#         else:
#             self.paths_illumination_all = None
#             self.illumination_name_to_path = {}
#             print("警告: 未提供Illumination数据路径")
#
#         # 获取Velocity数据路径并创建映射
#         if opt.get('dataroot_velocity'):
#             self.paths_velocity_all = self.get_mat_paths(opt['dataroot_velocity'])
#             self.velocity_name_to_path = {}
#             for path in self.paths_velocity_all:
#                 base_name = os.path.splitext(os.path.basename(path))[0]  # 例如: "example1"
#                 self.velocity_name_to_path[base_name] = path
#             print(f"Velocity数据加载: {len(self.paths_velocity_all)}个文件")
#         else:
#             self.paths_velocity_all = None
#             self.velocity_name_to_path = {}
#             print("警告: 未提供Velocity数据路径")
#
#         # 记录哪些HR数据有对应的先验信息
#         self.has_prior = {}
#         for h_name in self.h_name_to_path.keys():
#             # 先计算对应的LR/先验文件名
#             lr_prior_name = h_name + self.lr_suffix  # 例如: "case387_crossline_0" + "x2" = "case387_crossline_0x2"
#
#             has_illumination = lr_prior_name in self.illumination_name_to_path
#             has_velocity = lr_prior_name in self.velocity_name_to_path
#             self.has_prior[h_name] = has_illumination or has_velocity
#
#         print(f"数据集统计: 总样本数={len(self.paths_H)}, "
#               f"有先验信息的样本数={sum(self.has_prior.values())}")
#
#         # 在__init__方法最后添加
#         self.check_data_availability()
#
#         self.scale = opt.get('scale', 1)
#         # 在__init__方法最后添加数据验证
#         self.validate_dataset()
#
#     def validate_dataset(self):
#         """验证数据集中的尺寸关系"""
#         print("\n验证数据集尺寸关系:")
#
#         for i in range(min(20, len(self.paths_H))):  # 检查前20个样本
#             H_path = self.paths_H[i]
#             h_base_name = os.path.splitext(os.path.basename(H_path))[0]
#             lr_prior_name = h_base_name + self.lr_suffix
#
#             # 加载HR数据
#             img_H = self.load_seismic_data(H_path, self.key_H)
#
#             # 检查是否有对应的LR
#             if lr_prior_name in self.l_name_to_path:
#                 L_path = self.l_name_to_path[lr_prior_name]
#                 img_L = self.load_seismic_data(L_path, self.key_L)
#
#                 # 验证尺寸关系
#                 if img_H.shape[0] == img_L.shape[0] * self.sf and img_H.shape[1] == img_L.shape[1] * self.sf:
#                     print(f"  ✓ {h_base_name}: HR={img_H.shape[:2]}, LR={img_L.shape[:2]}")
#                 else:
#                     print(f"  ✗ {h_base_name}: HR={img_H.shape[:2]}, LR={img_L.shape[:2]} (不匹配!)")
#             else:
#                 print(f"  ? {h_base_name}: 无对应LR数据")
#
#     def get_base_name_from_path(self, path):
#         """从文件路径提取基本文件名（不含扩展名）"""
#         return os.path.splitext(os.path.basename(path))[0]
#
#     def get_corresponding_lr_name(self, h_base_name):
#         """根据HR文件名获取对应的LR文件名"""
#         return h_base_name + "x2"  # 根据你的命名规则
#
#     def check_data_availability(self):
#         """检查数据可用性并打印统计信息"""
#         print("=" * 60)
#         print("数据集统计信息:")
#         print(f"HR数据总数: {len(self.paths_H)}")
#
#         # 检查LR数据匹配
#         lr_matched = 0
#         lr_mismatched = []
#
#         for h_name in self.h_name_to_path.keys():
#             lr_prior_name = h_name + self.lr_suffix
#             if lr_prior_name in self.l_name_to_path:
#                 lr_matched += 1
#             else:
#                 lr_mismatched.append(h_name)
#
#         print(f"成功匹配的LR数据样本数: {lr_matched}/{len(self.paths_H)}")
#
#         # 打印匹配示例
#         if lr_matched > 0:
#             print("LR匹配示例:")
#             sample_h_names = list(self.h_name_to_path.keys())[:3]
#             for h_name in sample_h_names:
#                 lr_prior_name = h_name + self.lr_suffix
#                 if lr_prior_name in self.l_name_to_path:
#                     print(f"  ✓ {h_name} -> {lr_prior_name}")
#                 else:
#                     print(f"  ✗ {h_name} -> {lr_prior_name} (未找到)")
#
#         # 检查先验数据匹配
#         illum_matched = 0
#         vel_matched = 0
#
#         for h_name in self.h_name_to_path.keys():
#             lr_prior_name = h_name + self.lr_suffix
#             if lr_prior_name in self.illumination_name_to_path:
#                 illum_matched += 1
#             if lr_prior_name in self.velocity_name_to_path:
#                 vel_matched += 1
#
#         print(f"有Illumination数据的样本数: {illum_matched}/{len(self.paths_H)}")
#         print(f"有Velocity数据的样本数: {vel_matched}/{len(self.paths_H)}")
#
#         # 打印文件示例
#         print("\n实际文件示例:")
#         if self.paths_H:
#             hr_example = os.path.basename(self.paths_H[0])
#             print(f"HR文件示例: {hr_example}")
#             h_base = os.path.splitext(hr_example)[0]
#             expected_lr = h_base + self.lr_suffix + ".mat"
#             print(f"期望的LR/先验文件名: {expected_lr}")
#
#         if self.paths_L_all:
#             lr_example = os.path.basename(self.paths_L_all[0])
#             print(f"实际LR文件示例: {lr_example}")
#
#         if self.paths_illumination_all:
#             illum_example = os.path.basename(self.paths_illumination_all[0])
#             print(f"实际Illumination文件示例: {illum_example}")
#
#         if self.paths_velocity_all:
#             vel_example = os.path.basename(self.paths_velocity_all[0])
#             print(f"实际Velocity文件示例: {vel_example}")
#
#         # 如果有不匹配的情况，打印前几个
#         if lr_mismatched:
#             print(f"\n未匹配的HR文件示例 (共{len(lr_mismatched)}个):")
#             for h_name in lr_mismatched[:5]:
#                 print(f"  {h_name}")
#             if len(lr_mismatched) > 5:
#                 print(f"  ... 还有{len(lr_mismatched) - 5}个")
#
#         print("=" * 60)
#
#     def get_mat_paths(self, dataroot):
#         """Get paths to all .mat files in the directory"""
#         paths = []
#         if dataroot and os.path.exists(dataroot):
#             for fname in sorted(os.listdir(dataroot)):
#                 if fname.endswith('.mat'):
#                     paths.append(os.path.join(dataroot, fname))
#         return paths
#
#     def load_seismic_data(self, path, key):
#         """Load seismic data from .mat file"""
#         try:
#             data = sio.loadmat(path)
#
#             # 调试：打印文件中的键名
#             if self.opt.get('debug', False):
#                 print(f"DEBUG: 加载文件 {path}")
#                 print(f"  可用键名: {list(data.keys())}")
#                 print(f"  寻找键名: {key}")
#
#             # 尝试使用指定的键名获取数据
#             if key in data:
#                 seismic = data[key].astype(np.float32)
#                 if self.opt.get('debug', False):
#                     print(f"  使用键名 '{key}', 形状: {seismic.shape}")
#             else:
#                 # 如果指定的键名不存在，尝试使用第一个非内部变量
#                 for k in data.keys():
#                     if not k.startswith('__'):
#                         seismic = data[k].astype(np.float32)
#                         if self.opt.get('debug', False):
#                             print(f"  使用自动选择的键名 '{k}', 形状: {seismic.shape}")
#                         break
#                 else:
#                     raise ValueError(f"No valid data found in {path}")
#
#             # 确保数据是2D或3D的
#             if seismic.ndim == 2:
#                 seismic = seismic[..., np.newaxis]  # 添加通道维度: (H, W) -> (H, W, 1)
#             elif seismic.ndim == 3 and seismic.shape[2] > 3:
#                 # 如果通道数大于3，取前3个通道
#                 seismic = seismic[:, :, :3]
#
#             # 调试：最终形状
#             if self.opt.get('debug', False):
#                 print(f"  最终形状: {seismic.shape}")
#
#             return seismic
#         except Exception as e:
#             print(f"Error loading {path}: {e}")
#             raise
#
#     def __getitem__(self, index):
#         # ------------------------------------
#         # get H (target) seismic data
#         # ------------------------------------
#         H_path = self.paths_H[index]
#         h_base_name = os.path.splitext(os.path.basename(H_path))[0]  # 例如: "example1"
#         # # 添加调试信息
#         # if index < 10:  # 打印前10个样本的详细信息
#         #     print(f"\n{'=' * 60}")
#         #     print(f"样本 {index}: {h_base_name}")
#
#         img_H = self.load_seismic_data(H_path, self.key_H)
#         # if index < 10:
#         #     print(f"  HR形状: {img_H.shape}")
#         #     print(f"  HR路径: {H_path}")
#
#
#         # ------------------------------------
#         # 获取L (input) seismic data
#         # ------------------------------------
#         # 根据HR文件名构建对应的LR文件名
#         lr_prior_name = h_base_name + self.lr_suffix  # 例如: "case387_crossline_0" + "x2" = "case387_crossline_0x2"
#         # if index < 10:
#         #     print(f"  期望的LR文件名: {lr_prior_name}")
#
#         # 检查LR文件是否存在
#         has_lr = lr_prior_name in self.l_name_to_path
#
#         if has_lr:
#             L_path = self.l_name_to_path[lr_prior_name]
#             img_L = self.load_seismic_data(L_path, self.key_L)
#             # if index < 10:
#             #     print(f"  LR路径: {L_path}")
#             #     print(f"  LR形状: {img_L.shape}")
#             # 验证尺寸关系
#             if img_H.shape[0] != img_L.shape[0] * self.sf or img_H.shape[1] != img_L.shape[1] * self.sf:
#                 print( f"  ⚠️ 警告: HR/LR尺寸不匹配! HR={img_H.shape[:2]}, LR={img_L.shape[:2]}, 期望HR={img_L.shape[0] * self.sf}x{img_L.shape[1] * self.sf}")
#         else:
#             # 如果没有对应的LR数据，使用H作为L（或报错）
#             if self.opt.get('use_H_as_L_when_missing', False):
#                 # 如果HR是256x256，需要下采样到128x128作为LR
#                 if img_H.shape[0] == 256 and img_H.shape[1] == 256 and self.sf == 2:
#                     # 简单下采样
#                     import cv2
#                     img_L = cv2.resize(img_H[:, :, 0], (128, 128), interpolation=cv2.INTER_CUBIC)
#                     img_L = img_L[:, :, np.newaxis]
#                 else:
#                     img_L = img_H.copy()
#                 L_path = H_path
#                 print(f"警告: {h_base_name} 没有对应的LR数据，使用HR作为LR")
#             else:
#                 # 列出可用的LR文件供调试
#                 available_lr_files = list(self.l_name_to_path.keys())[:10]
#                 print(f"可用的LR文件示例: {available_lr_files}")
#                 raise FileNotFoundError(f"没有找到 {h_base_name} 对应的LR数据。"
#                                         f"期望的文件名: {lr_prior_name}.mat")
#             #     img_L = img_H.copy()
#             #     L_path = H_path
#             #     print(f"警告: {h_base_name} 没有对应的LR数据，使用HR作为LR")
#             # else:
#             #     # 列出可用的LR文件供调试
#             #     available_lr_files = list(self.l_name_to_path.keys())[:10]
#             #     print(f"可用的LR文件示例: {available_lr_files}")
#             #     raise FileNotFoundError(f"没有找到 {h_base_name} 对应的LR数据。"
#             #                             f"期望的文件名: {lr_prior_name}.mat")
#
#         # ------------------------------------
#         # 获取illumination数据（如果有）
#         # ------------------------------------
#         if lr_prior_name in self.illumination_name_to_path:
#             illumination_path = self.illumination_name_to_path[lr_prior_name]
#             img_illumination = self.load_seismic_data(illumination_path, self.key_illumination)
# #            if index < 10:
# #                print(f"  Illumination路径: {illumination_path}")
# #                print(f"  Illumination形状: {img_illumination.shape}")
#
#         else:
#             # 如果没有illumination数据，创建全1的矩阵
#             # 注意：应该是LR尺寸，不是HR尺寸！
#             img_illumination = np.ones((img_L.shape[0], img_L.shape[1], 1), dtype=np.float32)
#             illumination_path = f"dummy_illumination_{h_base_name}"
#             # if index < 10:
#             #     print(f"  创建虚拟illumination，形状: {img_illumination.shape}")
#
#
#         # ------------------------------------
#         # 获取velocity数据（如果有）
#         # ------------------------------------
#         if lr_prior_name in self.velocity_name_to_path:
#             velocity_path = self.velocity_name_to_path[lr_prior_name]
#             img_velocity = self.load_seismic_data(velocity_path, self.key_velocity)
#             # if index < 10:
#             #     print(f"  Velocity路径: {velocity_path}")
#             #     print(f"  Velocity形状: {img_velocity.shape}")
#         else:
#             # 如果没有velocity数据，创建全1的矩阵
#             # 如果没有velocity数据，创建与LR相同尺寸的全1矩阵
#             img_velocity = np.ones((img_L.shape[0], img_L.shape[1], 1), dtype=np.float32)
#             velocity_path = f"dummy_velocity_{h_base_name}"
#             # if index < 10:
#             #     print(f"  创建虚拟velocity，形状: {img_velocity.shape}")
#
#             # if index < 10:
#             #     print(f"  {'=' * 60}")
#
#         # ------------------------------------
#         # if train, get L/H patch pair
#         # ------------------------------------
#         # if self.opt['phase'] == 'train':
#         #     H, W, _ = img_L.shape
#         #     min_H = min(img_L.shape[0], img_illumination.shape[0], img_velocity.shape[0], img_H.shape[0])
#         #     min_W = min(img_L.shape[1], img_illumination.shape[1], img_velocity.shape[1], img_H.shape[1])
#         #
#         #     # # 确保所有图像尺寸一致
#         #     # H, W = img_L.shape[0], img_L.shape[1]
#         #     #
#         #     # # 获取最小公共尺寸
#         #     # min_H = min(img_L.shape[0], img_illumination.shape[0], img_velocity.shape[0], img_H.shape[0])
#         #     # min_W = min(img_L.shape[1], img_illumination.shape[1], img_velocity.shape[1], img_H.shape[1])
#         #
#         #     # 如果尺寸不匹配，调整到最小尺寸
#         #     if min_H < H or min_W < W:
#         #         H, W = min_H, min_W
#         #         img_L = img_L[:H, :W, :]
#         #         img_illumination = img_illumination[:H, :W, :]
#         #         img_velocity = img_velocity[:H, :W, :]
#         #         img_H = img_H[:H, :W, :]
#         #
#         #         # 随机裁剪patch
#         #     if H > self.patch_size and W > self.patch_size:
#         #         rnd_h = random.randint(0, H - self.patch_size)
#         #         rnd_w = random.randint(0, W - self.patch_size)
#         #
#         #         img_L = img_L[rnd_h:rnd_h + self.patch_size,
#         #                 rnd_w:rnd_w + self.patch_size, :]
#         #         img_illumination = img_illumination[rnd_h:rnd_h + self.patch_size,
#         #                            rnd_w:rnd_w + self.patch_size, :]
#         #         img_velocity = img_velocity[rnd_h:rnd_h + self.patch_size,
#         #                        rnd_w:rnd_w + self.patch_size, :]
#         #         img_H = img_H[rnd_h:rnd_h + self.patch_size,
#         #                 rnd_w:rnd_w + self.patch_size, :]
#         #     else:
#         #         # 如果图像小于patch size，进行填充或调整patch size
#         #         new_patch_size = min(H, W)
#         #         # 或者你可以选择填充图像到patch size
#         #         # 这里简单起见，我们调整patch size到图像尺寸
#         #         img_L = img_L[:new_patch_size, :new_patch_size, :]
#         #         img_illumination = img_illumination[:new_patch_size, :new_patch_size, :]
#         #         img_velocity = img_velocity[:new_patch_size, :new_patch_size, :]
#         #         img_H = img_H[:new_patch_size, :new_patch_size, :]
#         #
#         #         # 数据增强
#         #     if self.opt.get('use_flip', True):
#         #         mode = random.randint(0, 7)
#         #         img_L = util.augment_seismic(img_L, mode=mode)
#         #         img_illumination = util.augment_seismic(img_illumination, mode=mode)
#         #         img_velocity = util.augment_seismic(img_velocity, mode=mode)
#         #         img_H = util.augment_seismic(img_H, mode=mode)
#         #
#         #         # 添加噪声
#         #     if self.opt.get('use_noise', False):
#         #         noise_level = self.opt.get('noise_level', 0.01)
#         #         noise = np.random.normal(0, noise_level, img_L.shape)
#         #         img_L = img_L + noise
#
#         if self.opt['phase'] == 'train':
#             # 验证HR和LR的尺寸关系
#             if img_H.shape[0] != img_L.shape[0] * self.sf or img_H.shape[1] != img_L.shape[1] * self.sf:
#                 print(
#                     f"样本 {index}: 严重错误 - HR={img_H.shape[:2]}, LR={img_L.shape[:2]}, 期望HR={img_L.shape[0] * self.sf}x{img_L.shape[1] * self.sf}")
#                 raise ValueError(f"HR和LR尺寸不满足{self.sf}倍关系!")
#
#             # 修正的随机裁剪逻辑
#             if img_L.shape[0] > self.patch_size and img_L.shape[1] > self.patch_size:
#                 # 在LR图像上选择随机位置
#                 rnd_h = random.randint(0, img_L.shape[0] - self.patch_size)
#                 rnd_w = random.randint(0, img_L.shape[1] - self.patch_size)
#
#                 # 裁剪LR图像
#                 img_L = img_L[rnd_h:rnd_h + self.patch_size,
#                         rnd_w:rnd_w + self.patch_size, :]
#
#                 # 裁剪先验数据（与LR相同位置和大小）
#                 img_illumination = img_illumination[rnd_h:rnd_h + self.patch_size,
#                                    rnd_w:rnd_w + self.patch_size, :]
#                 img_velocity = img_velocity[rnd_h:rnd_h + self.patch_size,
#                                rnd_w:rnd_w + self.patch_size, :]
#
#                 # 计算HR的裁剪位置和大小（乘以缩放因子）
#                 hr_rnd_h = rnd_h * self.sf
#                 hr_rnd_w = rnd_w * self.sf
#                 hr_patch_size = self.patch_size * self.sf
#
#                 # 裁剪HR图像
#                 img_H = img_H[hr_rnd_h:hr_rnd_h + hr_patch_size,
#                         hr_rnd_w:hr_rnd_w + hr_patch_size, :]
#             else:
#                 # 如果图像小于patch size，使用整个图像
#                 # 但需要确保LR和HR保持2倍关系
#                 lr_size = min(img_L.shape[0], img_L.shape[1])
#                 hr_size = lr_size * self.sf
#
#                 img_L = img_L[:lr_size, :lr_size, :]
#                 img_illumination = img_illumination[:lr_size, :lr_size, :]
#                 img_velocity = img_velocity[:lr_size, :lr_size, :]
#                 img_H = img_H[:hr_size, :hr_size, :]
#
#             # 数据增强
#             if self.opt.get('use_flip', True):
#                 mode = random.randint(0, 7)
#                 img_L = util.augment_seismic(img_L, mode=mode)
#                 img_illumination = util.augment_seismic(img_illumination, mode=mode)
#                 img_velocity = util.augment_seismic(img_velocity, mode=mode)
#                 img_H = util.augment_seismic(img_H, mode=mode)
#
#             # 添加噪声
#             if self.opt.get('use_noise', False):
#                 noise_level = self.opt.get('noise_level', 0.01)
#                 noise = np.random.normal(0, noise_level, img_L.shape)
#                 img_L = img_L + noise
#
#
#
#
#
#         # # 在转换为张量之前，添加尺寸打印
#         # print(f"样本 {index}: LR形状={img_L.shape}, HR形状={img_H.shape}, "
#         #       f"缩放因子={self.sf}, 期望HR形状=({img_L.shape[0] * self.sf}, {img_L.shape[1] * self.sf})")
#
#         # 验证尺寸比例
#         if img_H.shape[0] != img_L.shape[0] * self.sf or img_H.shape[1] != img_L.shape[1] * self.sf:
#             print(f"⚠️ 尺寸不匹配警告: LR={img_L.shape[:2]}, HR={img_H.shape[:2]}, 不是{self.sf}倍关系!")
#
#         # ------------------------------------
#         # Convert to PyTorch tensor
#         # ------------------------------------
#         # HWC to CHW: (H, W, C) -> (C, H, W)
#         img_H = torch.from_numpy(img_H.transpose(2, 0, 1).copy()).float()
#         img_L = torch.from_numpy(img_L.transpose(2, 0, 1).copy()).float()
#         img_illumination = torch.from_numpy(img_illumination.transpose(2, 0, 1).copy()).float()
#         img_velocity = torch.from_numpy(img_velocity.transpose(2, 0, 1).copy()).float()
#
#         # 创建标志位，指示是否有先验信息
#         has_prior_tensor = torch.tensor([float(self.has_prior[h_base_name])])
#
#         return {'L': img_L,
#                 'illumination': img_illumination,
#                 'velocity': img_velocity,
#                 'H': img_H,
#                 'L_path': L_path,
#                 'illumination_path': illumination_path,
#                 'velocity_path': velocity_path,
#                 'H_path': H_path,
#                 'has_prior': has_prior_tensor,
#                 'base_name': h_base_name,
#                 'lr_prior_name': lr_prior_name  # 添加这个字段用于调试
#                 }
#
#     def __len__(self):
#         return len(self.paths_H)





# # 64x64
# # -*-coding:utf-8-*-
# import random
# import numpy as np
# import torch
# import torch.utils.data as data
# import utils.utils_image as util
# import scipy.io as sio
# import os
#
#
# class DatasetSR(data.Dataset):
#     '''
#     地震数据加载器，适配每个 .mat 文件包含多个小块（如 16×64×64）的情况。
#     每个小块作为一个独立样本。
#     支持文件名前缀替换（如 refl -> rtm / hessian / vpsm）。
#     '''
#
#     def __init__(self, opt):
#         super(DatasetSR, self).__init__()
#         self.opt = opt
#         self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
#         self.sf = opt['scale'] if opt['scale'] else 4
#         self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
#         self.L_size = self.patch_size // self.sf
#
#         # 每个文件包含的小块数量（默认16）
#         self.num_patches_per_file = opt.get('num_patches_per_file', 16)
#
#         # 定义不同数据类型的键名
#         self.key_H = opt.get('key_H', 'refl')
#         self.key_L = opt.get('key_L', 'rtm')
#         self.key_illumination = opt.get('key_illumination', 'hessian')
#         self.key_velocity = opt.get('key_velocity', 'vpsm')
#
#         # LR后缀（通常用于超分辨任务，若文件名无后缀可设为空字符串）
#         self.lr_suffix = opt.get('lr_suffix', 'x2')
#
#         # ---------- 新增：文件名替换规则（可从JSON配置读取） ----------
#         self.lr_replace_from = opt.get('lr_replace_from', 'refl')
#         self.lr_replace_to = opt.get('lr_replace_to', 'rtm')
#         self.illumination_replace_from = opt.get('illumination_replace_from', 'refl')
#         self.illumination_replace_to = opt.get('illumination_replace_to', 'hessian')
#         self.velocity_replace_from = opt.get('velocity_replace_from', 'refl')
#         self.velocity_replace_to = opt.get('velocity_replace_to', 'vpsm')
#         # -------------------------------------------------------------
#
#         # 获取所有数据路径
#         self.paths_H = self.get_mat_paths(opt['dataroot_H'])
#         assert self.paths_H, 'Error: H path is empty.'
#
#         # 创建文件名到路径的映射（不含扩展名）
#         self.h_name_to_path = {}
#         self.h_name_to_idx = {}
#         for idx, path in enumerate(self.paths_H):
#             base_name = os.path.splitext(os.path.basename(path))[0]
#             self.h_name_to_path[base_name] = path
#             self.h_name_to_idx[base_name] = idx
#
#         # 获取LR数据路径并创建映射
#         if opt.get('dataroot_L'):
#             self.paths_L_all = self.get_mat_paths(opt['dataroot_L'])
#             self.l_name_to_path = {}
#             for path in self.paths_L_all:
#                 base_name = os.path.splitext(os.path.basename(path))[0]
#                 self.l_name_to_path[base_name] = path
#             print(f"LR数据加载: {len(self.paths_L_all)}个文件")
#         else:
#             self.paths_L_all = None
#             self.l_name_to_path = {}
#             print("警告: 未提供LR数据路径")
#
#         # 获取Illumination数据路径并创建映射
#         if opt.get('dataroot_illumination'):
#             self.paths_illumination_all = self.get_mat_paths(opt['dataroot_illumination'])
#             self.illumination_name_to_path = {}
#             for path in self.paths_illumination_all:
#                 base_name = os.path.splitext(os.path.basename(path))[0]
#                 self.illumination_name_to_path[base_name] = path
#             print(f"Illumination数据加载: {len(self.paths_illumination_all)}个文件")
#         else:
#             self.paths_illumination_all = None
#             self.illumination_name_to_path = {}
#             print("警告: 未提供Illumination数据路径")
#
#         # 获取Velocity数据路径并创建映射
#         if opt.get('dataroot_velocity'):
#             self.paths_velocity_all = self.get_mat_paths(opt['dataroot_velocity'])
#             self.velocity_name_to_path = {}
#             for path in self.paths_velocity_all:
#                 base_name = os.path.splitext(os.path.basename(path))[0]
#                 self.velocity_name_to_path[base_name] = path
#             print(f"Velocity数据加载: {len(self.paths_velocity_all)}个文件")
#         else:
#             self.paths_velocity_all = None
#             self.velocity_name_to_path = {}
#             print("警告: 未提供Velocity数据路径")
#
#         # 使用新方法计算 has_prior
#         self.has_prior = {}
#         for h_name in self.h_name_to_path.keys():
#             illum_name = self._get_corresponding_name(h_name, 'illumination')
#             vel_name = self._get_corresponding_name(h_name, 'velocity')
#             self.has_prior[h_name] = (illum_name in self.illumination_name_to_path) or \
#                                      (vel_name in self.velocity_name_to_path)
#
#         print(f"数据集统计: 总文件数={len(self.paths_H)}, "
#               f"每个文件包含小块数={self.num_patches_per_file}, "
#               f"总样本数={len(self.paths_H) * self.num_patches_per_file}, "
#               f"有先验信息的文件数={sum(self.has_prior.values())}")
#
#         self.check_data_availability()
#         self.validate_dataset()
#
#     # ------------------------------------------------------------
#     # 新增辅助方法：根据HR基础名获取对应模态的文件名
#     # ------------------------------------------------------------
#     def _get_corresponding_name(self, h_base_name, data_type):
#         """
#         根据HR文件名（不含扩展名）和数据类型，返回可能的对应文件名。
#         优先尝试添加后缀（lr_suffix），若不存在则使用替换规则。
#         data_type: 'L', 'illumination', 'velocity'
#         """
#         if data_type == 'L':
#             name_with_suffix = h_base_name + self.lr_suffix
#             if name_with_suffix in self.l_name_to_path:
#                 return name_with_suffix
#             # 否则使用替换规则
#             replaced = h_base_name.replace(self.lr_replace_from, self.lr_replace_to)
#             return replaced
#         elif data_type == 'illumination':
#             name_with_suffix = h_base_name + self.lr_suffix
#             if name_with_suffix in self.illumination_name_to_path:
#                 return name_with_suffix
#             replaced = h_base_name.replace(self.illumination_replace_from, self.illumination_replace_to)
#             return replaced
#         elif data_type == 'velocity':
#             name_with_suffix = h_base_name + self.lr_suffix
#             if name_with_suffix in self.velocity_name_to_path:
#                 return name_with_suffix
#             replaced = h_base_name.replace(self.velocity_replace_from, self.velocity_replace_to)
#             return replaced
#         else:
#             raise ValueError(f"Unknown data_type: {data_type}")
#
#     # ------------------------------------------------------------
#     # 验证数据集（原方法，已修改使用新方法）
#     # ------------------------------------------------------------
#     def validate_dataset(self):
#         print("\n验证数据集尺寸关系:")
#         for i in range(min(5, len(self.paths_H))):
#             H_path = self.paths_H[i]
#             h_base_name = os.path.splitext(os.path.basename(H_path))[0]
#             lr_prior_name = self._get_corresponding_name(h_base_name, 'L')
#             img_H = self.load_patch_from_mat(H_path, self.key_H, 0)
#             if lr_prior_name in self.l_name_to_path:
#                 L_path = self.l_name_to_path[lr_prior_name]
#                 img_L = self.load_patch_from_mat(L_path, self.key_L, 0)
#                 if img_H.shape[0] == img_L.shape[0] * self.sf and img_H.shape[1] == img_L.shape[1] * self.sf:
#                     print(f"  ✓ {h_base_name}: HR={img_H.shape[:2]}, LR={img_L.shape[:2]}")
#                 else:
#                     print(f"  ✗ {h_base_name}: HR={img_H.shape[:2]}, LR={img_L.shape[:2]} (尺寸不匹配!)")
#             else:
#                 print(f"  ? {h_base_name}: 无对应LR数据")
#
#     # ------------------------------------------------------------
#     # 其他原有方法（保持不变）
#     # ------------------------------------------------------------
#     def get_base_name_from_path(self, path):
#         return os.path.splitext(os.path.basename(path))[0]
#
#     def get_corresponding_lr_name(self, h_base_name):
#         return h_base_name + self.lr_suffix
#
#     def check_data_availability(self):
#         print("=" * 60)
#         print("数据集统计信息:")
#         print(f"HR文件总数: {len(self.paths_H)}")
#
#         lr_matched = 0
#         lr_mismatched = []
#         for h_name in self.h_name_to_path.keys():
#             lr_prior_name = self._get_corresponding_name(h_name, 'L')
#             if lr_prior_name in self.l_name_to_path:
#                 lr_matched += 1
#             else:
#                 lr_mismatched.append(h_name)
#         print(f"成功匹配的LR文件数: {lr_matched}/{len(self.paths_H)}")
#
#         if lr_matched > 0:
#             print("LR匹配示例:")
#             sample_h_names = list(self.h_name_to_path.keys())[:3]
#             for h_name in sample_h_names:
#                 lr_prior_name = self._get_corresponding_name(h_name, 'L')
#                 if lr_prior_name in self.l_name_to_path:
#                     print(f"  ✓ {h_name} -> {lr_prior_name}")
#                 else:
#                     print(f"  ✗ {h_name} -> {lr_prior_name} (未找到)")
#
#         illum_matched = 0
#         vel_matched = 0
#         for h_name in self.h_name_to_path.keys():
#             illum_name = self._get_corresponding_name(h_name, 'illumination')
#             vel_name = self._get_corresponding_name(h_name, 'velocity')
#             if illum_name in self.illumination_name_to_path:
#                 illum_matched += 1
#             if vel_name in self.velocity_name_to_path:
#                 vel_matched += 1
#         print(f"有Illumination数据的文件数: {illum_matched}/{len(self.paths_H)}")
#         print(f"有Velocity数据的文件数: {vel_matched}/{len(self.paths_H)}")
#
#         print("\n实际文件示例:")
#         if self.paths_H:
#             hr_example = os.path.basename(self.paths_H[0])
#             print(f"HR文件示例: {hr_example}")
#             h_base = os.path.splitext(hr_example)[0]
#             expected_lr = self._get_corresponding_name(h_base, 'L') + ".mat"
#             print(f"期望的LR/先验文件名: {expected_lr}")
#
#         if self.paths_L_all:
#             lr_example = os.path.basename(self.paths_L_all[0])
#             print(f"实际LR文件示例: {lr_example}")
#
#         if self.paths_illumination_all:
#             illum_example = os.path.basename(self.paths_illumination_all[0])
#             print(f"实际Illumination文件示例: {illum_example}")
#
#         if self.paths_velocity_all:
#             vel_example = os.path.basename(self.paths_velocity_all[0])
#             print(f"实际Velocity文件示例: {vel_example}")
#
#         if lr_mismatched:
#             print(f"\n未匹配的HR文件示例 (共{len(lr_mismatched)}个):")
#             for h_name in lr_mismatched[:5]:
#                 print(f"  {h_name}")
#             if len(lr_mismatched) > 5:
#                 print(f"  ... 还有{len(lr_mismatched) - 5}个")
#         print("=" * 60)
#
#     def get_mat_paths(self, dataroot):
#         paths = []
#         if dataroot and os.path.exists(dataroot):
#             for fname in sorted(os.listdir(dataroot)):
#                 if fname.endswith('.mat'):
#                     paths.append(os.path.join(dataroot, fname))
#         return paths
#
#     def load_patch_from_mat(self, path, key, patch_idx):
#         try:
#             data = sio.loadmat(path)
#             if key in data:
#                 arr = data[key].astype(np.float32)
#             else:
#                 for k in data.keys():
#                     if not k.startswith('__'):
#                         arr = data[k].astype(np.float32)
#                         break
#                 else:
#                     raise ValueError(f"No valid data found in {path}")
#
#             if arr.ndim == 3:
#                 if arr.shape[0] <= patch_idx:
#                     raise IndexError(f"patch_idx {patch_idx} 超出文件 {path} 中的块数 {arr.shape[0]}")
#                 patch = arr[patch_idx]
#                 patch = patch[..., np.newaxis]
#             elif arr.ndim == 4:
#                 if arr.shape[0] <= patch_idx:
#                     raise IndexError(f"patch_idx {patch_idx} 超出文件 {path} 中的块数 {arr.shape[0]}")
#                 patch = arr[patch_idx]
#             else:
#                 raise ValueError(f"Unsupported array shape {arr.shape} in {path}")
#
#             return patch
#         except Exception as e:
#             print(f"Error loading {path} (patch_idx={patch_idx}): {e}")
#             raise
#
#     def load_seismic_data(self, path, key):
#         try:
#             data = sio.loadmat(path)
#             if key in data:
#                 seismic = data[key].astype(np.float32)
#             else:
#                 for k in data.keys():
#                     if not k.startswith('__'):
#                         seismic = data[k].astype(np.float32)
#                         break
#                 else:
#                     raise ValueError(f"No valid data found in {path}")
#
#             if seismic.ndim == 2:
#                 seismic = seismic[..., np.newaxis]
#             elif seismic.ndim == 3 and seismic.shape[2] > 3:
#                 seismic = seismic[:, :, :3]
#             return seismic
#         except Exception as e:
#             print(f"Error loading {path}: {e}")
#             raise
#
#     # ------------------------------------------------------------
#     # 核心：__getitem__
#     # ------------------------------------------------------------
#     def __getitem__(self, index):
#         file_idx = index // self.num_patches_per_file
#         patch_idx = index % self.num_patches_per_file
#
#         H_path = self.paths_H[file_idx]
#         h_base_name = os.path.splitext(os.path.basename(H_path))[0]
#
#         img_H = self.load_patch_from_mat(H_path, self.key_H, patch_idx)
#
#         # 使用新方法获取LR文件名
#         lr_prior_name = self._get_corresponding_name(h_base_name, 'L')
#         if lr_prior_name in self.l_name_to_path:
#             L_path = self.l_name_to_path[lr_prior_name]
#             img_L = self.load_patch_from_mat(L_path, self.key_L, patch_idx)
#         else:
#             if self.opt.get('use_H_as_L_when_missing', False):
#                 img_L = img_H.copy()
#                 L_path = H_path
#                 print(f"警告: {h_base_name} 没有对应的LR数据，使用HR作为LR")
#             else:
#                 raise FileNotFoundError(f"没有找到 {h_base_name} 对应的LR数据。期望的文件名: {lr_prior_name}.mat")
#
#         # illumination
#         illum_name = self._get_corresponding_name(h_base_name, 'illumination')
#         if illum_name in self.illumination_name_to_path:
#             illumination_path = self.illumination_name_to_path[illum_name]
#             img_illumination = self.load_patch_from_mat(illumination_path, self.key_illumination, patch_idx)
#         else:
#             img_illumination = np.ones((img_L.shape[0], img_L.shape[1], 1), dtype=np.float32)
#             illumination_path = f"dummy_illumination_{h_base_name}"
#
#         # velocity
#         vel_name = self._get_corresponding_name(h_base_name, 'velocity')
#         if vel_name in self.velocity_name_to_path:
#             velocity_path = self.velocity_name_to_path[vel_name]
#             img_velocity = self.load_patch_from_mat(velocity_path, self.key_velocity, patch_idx)
#         else:
#             img_velocity = np.ones((img_L.shape[0], img_L.shape[1], 1), dtype=np.float32)
#             velocity_path = f"dummy_velocity_{h_base_name}"
#
#         # 数据增强
#         if self.opt['phase'] == 'train':
#             if img_L.shape[0] > self.patch_size and img_L.shape[1] > self.patch_size:
#                 rnd_h = random.randint(0, img_L.shape[0] - self.patch_size)
#                 rnd_w = random.randint(0, img_L.shape[1] - self.patch_size)
#
#                 img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
#                 img_illumination = img_illumination[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
#                 img_velocity = img_velocity[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
#
#                 hr_rnd_h = rnd_h * self.sf
#                 hr_rnd_w = rnd_w * self.sf
#                 hr_patch_size = self.patch_size * self.sf
#                 img_H = img_H[hr_rnd_h:hr_rnd_h + hr_patch_size, hr_rnd_w:hr_rnd_w + hr_patch_size, :]
#             else:
#                 lr_size = min(img_L.shape[0], img_L.shape[1])
#                 hr_size = lr_size * self.sf
#                 img_L = img_L[:lr_size, :lr_size, :]
#                 img_illumination = img_illumination[:lr_size, :lr_size, :]
#                 img_velocity = img_velocity[:lr_size, :lr_size, :]
#                 img_H = img_H[:hr_size, :hr_size, :]
#
#             if self.opt.get('use_flip', True):
#                 mode = random.randint(0, 7)
#                 img_L = util.augment_seismic(img_L, mode=mode)
#                 img_illumination = util.augment_seismic(img_illumination, mode=mode)
#                 img_velocity = util.augment_seismic(img_velocity, mode=mode)
#                 img_H = util.augment_seismic(img_H, mode=mode)
#
#             if self.opt.get('use_noise', False):
#                 noise_level = self.opt.get('noise_level', 0.01)
#                 noise = np.random.normal(0, noise_level, img_L.shape)
#                 img_L = img_L + noise
#
#         # 验证尺寸比例
#         if img_H.shape[0] != img_L.shape[0] * self.sf or img_H.shape[1] != img_L.shape[1] * self.sf:
#             print(f"⚠️ 尺寸不匹配警告: LR={img_L.shape[:2]}, HR={img_H.shape[:2]}, 不是{self.sf}倍关系!")
#
#         # 转换为张量 (C, H, W)
#         img_H = torch.from_numpy(img_H.transpose(2, 0, 1).copy()).float()
#         img_L = torch.from_numpy(img_L.transpose(2, 0, 1).copy()).float()
#         img_illumination = torch.from_numpy(img_illumination.transpose(2, 0, 1).copy()).float()
#         img_velocity = torch.from_numpy(img_velocity.transpose(2, 0, 1).copy()).float()
#
#         has_prior_tensor = torch.tensor([float(self.has_prior[h_base_name])])
#
#         return {
#             'L': img_L,
#             'illumination': img_illumination,
#             'velocity': img_velocity,
#             'H': img_H,
#             'L_path': L_path,
#             'illumination_path': illumination_path,
#             'velocity_path': velocity_path,
#             'H_path': H_path,
#             'has_prior': has_prior_tensor,
#             'base_name': h_base_name,
#             'lr_prior_name': lr_prior_name,
#             'patch_idx': patch_idx
#         }
#
#     def __len__(self):
#         return len(self.paths_H) * self.num_patches_per_file