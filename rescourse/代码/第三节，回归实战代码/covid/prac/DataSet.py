import torch
from torch.utils.data import Dataset
import numpy as np


class myDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

X = np.random.rand(100,10)
Y = np.random.randint(0,2,100)

# dataset = myDataset(X, Y)
# print(len(dataset))
# print(dataset[1])

x1 = np.array([[1,2,3],[4,5,6],[7,8,9],[11,12,13]])
print(x1)
print(x1[1:])
print(x1[1:,1:])