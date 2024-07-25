import numpy as np
import os

#修改npy文件的内容

# 获取文件夹内所有npy文件
folder = r"E:\yinda\cgan\data\8\sub-002\eeg\npy"
output_path = r"E:\yinda\cgan\data\8\sub-002\eeg\npyfill"
files = [f for f in os.listdir(folder) if f.endswith('.npy')]

# 找到第一维度最长的长度
max_length = 0
for file in files:
    data = np.load(os.path.join(folder, file))
    if data.shape[0] > max_length:
        max_length = data.shape[0]

# 更新所有npy文件的尺寸
for file in files:
    file_path = os.path.join(folder, file)
    data = np.load(file_path)
    # 如果当前npy文件的第一维度小于最大长度，则进行填充操作
    if data.shape[0] < max_length:
        padding_shape = (max_length - data.shape[0],) + (data.shape[1],)
        padding_data = np.zeros(padding_shape)
        # 在原有数据后面填充零，使其长度达到最大长度
        data_padded = np.concatenate((data, padding_data), axis=0)
    else:
        data_padded = data

    # 将第二维度改为128（如果已经是128，则不会影响原始数据）
    new_shape = (max_length, 128) + data_padded.shape[2:]
    data_resized = np.resize(data_padded, new_shape)
    data_resized = np.transpose(data_resized)
    file_name=os.path.join(output_path,file)
    # 将处理后的数据保存回原文件
    np.save(file_name, data_resized)