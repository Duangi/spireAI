import torch
print(f"Python版本：{torch.sys.version.split()[0]}")  # 输出3.8.x
print(f"CUDA是否可用：{torch.cuda.is_available()}")   # 输出True
print(f"CUDA版本：{torch.version.cuda}")             # 输出13.0 