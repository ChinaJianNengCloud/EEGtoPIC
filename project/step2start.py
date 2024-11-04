import subprocess

# 将原始的命令行字符串分割成一个列表形式，以便 subprocess 处理。
command = [
    "python", "code/eeg_ldm.py",
    "--dataset", "EEG",
    "--num_epoch" ,"300" ,
    "--batch_size" ,"6",
    "--pretrain_mbm_path", "18encodedecode1.3/eeg_pretrain/20-05-2024-19-14-07/checkpoints/checkpoint.pth",
    "--eval_avg","True"
]

# 使用 subprocess.run() 运行命令。
subprocess.run(command)
