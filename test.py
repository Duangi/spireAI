import torch
device = torch.device('cuda')
a = torch.randn(2000,2000, device=device)
for _ in range(5000):
    a = a.matmul(a)
torch.cuda.synchronize()
print("done")