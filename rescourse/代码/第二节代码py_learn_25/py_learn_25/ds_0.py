import torch
import torch.nn as nn

y = torch.tensor([11.3, 23, 20], dtype=float)
soft = nn.Softmax(dim=-1)
print(soft(y))