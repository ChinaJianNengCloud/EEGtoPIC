import subprocess

# 将原始的命令行字符串分割成一个列表形式，以便 subprocess 处理。
command = [
    "python", "code/stageA1_eeg_pretrain.py",
    "--mask_ratio", "0.75",
    "--num_epoch" ,"600" ,
    "--batch_size" ,"64",
    "--root_path","./"
]

# 使用 subprocess.run() 运行命令。
subprocess.run(command)