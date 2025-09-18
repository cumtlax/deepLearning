import torch
import torch.nn as nn
#
# X = torch.tensor(2.0, requires_grad=True)
# Y = X**2 + 3*X + 1
# Y.backward()
# print(X.grad)
# print(torch.cuda.is_available())
#
# a = torch.arange(6)
# print(a)
# b = a.view(2, 3)
# print(b)

x = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(x)
soft = nn.Softmax(dim=1)
print(soft(x))