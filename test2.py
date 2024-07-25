import torch

#加载.pth文件
model_weights=torch.load(r'G:\yinda\dreamdiffusion\8\2pth\yinda.pth',map_location=torch.device('cpu'))
aaaa=0
#在此之后，你可以根据你加载的权重初始化模型，并使用这些权重进行推断或微调
for i in range(len(model_weights['dataset'])):
    tensor=model_weights['dataset'][i]['eeg']
    if torch.is_tensor(tensor) and tensor.dim() == 2 and tensor.size(0)>=2 and tensor.size(1)>=2:

        print(f"Variableatindex{i}isa2Dtensor.")
    else:
        b=i
        aaaa = aaaa+1
        print(f"Variableatindex{i}isnota2Dtensor.")
print('a')