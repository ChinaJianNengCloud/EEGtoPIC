import numpy as np
from sklearn.model_selection import train_test_split
import torch
##制作模型需要的分割文件

import os

def count_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

directory_path = r"G:\yinda\dreamdiffusion\18\1.3npy"  # 替换为你的文件夹路径
file_count = count_files(directory_path)
print("文件夹中的文件数量为:", file_count)
# 假设你有N个样本，所以你的索引将从0到N-1
N = file_count  # 改为你实际的样本数量
indices = np.arange(N)

# 将索引分割为训练集和其余部分（测试集 + 验证集）
train_indices, other_indices = train_test_split(indices, test_size=0.4, random_state=42)

# 再将其余部分划分为测试集和验证集
val_indices, test_indices = train_test_split(other_indices, test_size=0.5, random_state=42)

train_indices=train_indices.tolist()
val_indices=val_indices.tolist()
test_indices= test_indices.tolist()
# 保存划分好的索引到 .pth 文件中
torch.save({'splits':[{'train': train_indices,
    'val': val_indices,
    'test': test_indices}]
}, r'G:\yinda\dreamdiffusion\18\3.3split\all_dataset_split.pth')