import torch
import numpy as np

# list1 = [
#         [1,2,3,4,5],
#         [6,7,8,9,10],
#         [11,12,13,14,15]
#     ]
# print(list1)
# array = np.array(list1)
# tensor1 = torch.tensor(list1)
# print(tensor1)
# print(array)
#
#
x = torch.tensor(3.0)
x.requires_grad_(True)
y = x**2
y.backward()
print(x)
#
# x = x.detach()
# x.grad = torch.tensor(0.0)
# y2 = x**2
# y2.backward()
# print(y2)
# print(x.grad)  #2x

#创建张量

# tensor1 = torch.ones((100,4))
# print(tensor1)
# tensor2 = torch.zeros((10,4))
# print(tensor2)

# tensor3 = torch.normal(10, 1, (3,10, 4))
# print(tensor3)


#求和
tensor1 = torch.ones((10,4))
# print(tensor1)
#
# sum1 = torch.sum(tensor1, dim=1, keepdim=True)
# print(sum1)
print(tensor1.shape)



