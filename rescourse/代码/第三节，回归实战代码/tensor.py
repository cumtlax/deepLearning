import torch

x = torch.tensor(2.0)
print(x)
print(x.cpu().item())