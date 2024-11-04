import os
import numpy as np
from PIL import Image
import torch
##制作模型需要的eeg格式
# 输入路径
npy_folder = r'G:\yinda\dreamdiffusion\18\1.3npy'  # 改为你的 .npy 文件所在目录
img_folder = r'G:\yinda\183\18\stimuli'  # 改为你的 image 文件所在目录

# 创建空字典来储存数据和图像
data_dict = {'dataset': [], 'images': []}

for filename in os.listdir(img_folder):
    basename, ext = os.path.splitext(filename)
    data_dict['images'].append(basename)

# 遍历 .npy 和图像文件
for filename in os.listdir(npy_folder):
    basename, ext = os.path.splitext(filename)
    if ext == '.npy':
        npy_path = os.path.join(npy_folder, filename)
        basename = basename.split('-')
        img_name = basename[0]  # 获取图像名（不含后缀）

        # 在 images 列表中找到对应图像的索引
        img_index = data_dict['images'].index(img_name)

        # 加载 .npy 文件
        eeg_data = np.load(npy_path)
        eeg_data = torch.from_numpy(eeg_data)

        # 将它们添加到列表中
        data_dict['dataset'].append({'eeg': eeg_data, 'image': img_index, 'subject': 999})

# 将数据字典保存为 .pth 文件
torch.save(data_dict, r'G:\yinda\dreamdiffusion\18\2.3pth\yinda.pth')  # 改为你想要的输出文件名