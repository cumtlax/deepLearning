import numpy
import numpy as np
import torch

#tensor
# list1 = [
#         [1,2,3,4,5],
#         [6,7,8,9,10],
#         [11,12,13,14,15]
#     ]
# print(list1)
#
# array = np.array(list1)  #把list1转化为矩阵
# print(array)
#
# #矩阵的操作
# array2 = np.array(list1)
# print(array2)
#
# array3 = np.concatenate((array, array2), axis=0)
# print(array3)

list1 = [
        [1,2,3,4,5],
        [6,7,8,9,10],
        [11,12,13,14,15]
    ]
print(list1)

array = np.array(list1)  #把list1转化为矩阵
#切片
print(list1[1:3])
print(array[:, :])
#跳着切
idx = [1, 3]
print(array[:, idx])

