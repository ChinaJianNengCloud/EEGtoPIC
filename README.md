# 原项目网址
这个项目网址https://github.com/bbaaii/DreamDiffusion


# 项目介绍
这个项目是将受到图片刺激EEG的脑电生成出同类别的图片。
有两个步骤

第一步，训练eeg数据的编解码器。运行step1start.py

第二步，根据第一步训练好的eeg数据的编解码器和图片与微调stablediffusion模型。运行step2start.py

# 数据的介绍
使用的数据集有两个，一个是8数据集，另一个是18数据集。目前数据已经上传到百度网盘。
数据的存储格式是npy，每个npy需要包含'eeg': eeg_data, 'image': img_index, 'subject': 999
其中eeg_data为切片的二维eeg数据，img_index为图片路径的索引，subject为这个数据集的标志

最后在dataset中会将eeg时间不足的进行插值，通道不足的进行复制，过多的进行截取，最终得到的通道数*时间是128x512。


数据集8的刺激数据是人头像，有450张。有两个切片模型。
切片模型1，共有15921个npy数据。

切片模型2(切片要求更加严格)，共有10617个npy数据。


数据集18的刺激数据是各种类别的图片，有216个。有四个切片模型
切片模型1,259192个。

切片模型2,130240个。

切片模型3,110507个。

切片模型4,1625个。

