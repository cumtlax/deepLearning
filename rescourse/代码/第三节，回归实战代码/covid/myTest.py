import csv
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
# 导入 Python 内置的 CSV 模块，用于读写 .csv 文件（逗号分隔值文件）。
# import csv
# 在机器学习项目中，通常先用 Pandas 进行数据探索与预处理。
# 提供 DataFrame 对象（表格型数据结构），数据分析和操作
# 数据清洗、筛选、合并、分组统计
# 支持从 CSV、Excel、数据库等多种格式加载数据
import pandas as pd
# Dataset
# 是一个抽象类，你需要继承它来创建自己的数据集。
# 必须实现两个方法：
# __len__(self)：返回数据集大小
# __getitem__(self, idx)：根据索引返回一条样本
# DataLoader
# 接收 Dataset 实例，提供批量加载（batching）、打乱顺序（shuffling）、多进程加载等功能。
# 训练模型时，通常将 Dataset 包装进 DataLoader。
from torch.utils.data import DataLoader, Dataset


# 定义数据集
class CovidDataset(Dataset):
    def __init__(self, file_path, mode):
        with open(file_path, "r") as f:
            # 这里转化为列表是为了后面的操作
            origin_data = list(csv.reader(f))
            temp_data = np.array(origin_data)
            csv_data = temp_data[1:, 1:].astype(float)
            # 区分csv_data的下标,indices表示的下表
            if mode == "train":
                indices = [i for i in range(len(csv_data)) if i % 5 != 0]
                self.y = torch.tensor(csv_data[indices, -1])
                data = torch.tensor(csv_data[indices, :-1])
            elif mode == "val":
                indices = [i for i in range(len(csv_data)) if i % 5 == 0]
                self.y = torch.tensor(csv_data[indices, -1])
                data = torch.tensor(csv_data[indices, :-1])
            else:
                indices = [i for i in range(len(csv_data))]
                data = torch.tensor(csv_data[indices])

            # 转化为tensor类型，支持后续的运算
            self.data = (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True)
            self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode != "test":
            return self.data[idx].float(), self.y[idx].float()
        else:
            return self.data[idx].float()


train_file = "covid.train.csv"
test_file = "covid.test.csv"

train_dataset = CovidDataset(train_file, "train")
val_dataset = CovidDataset(train_file, "val")
test_dataset = CovidDataset(test_file, "test")

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class MyModel(nn.Module):
    def __init__(self, inDim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(inDim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        if len(x.size()) > 1:
            return x.squeeze(1)
        return x


def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = model.to(device)

    plt_train_loss = []
    plt_val_loss = []

    min_val_loss = 99999

    for epoch in range(epochs):
        # 每一轮的loss
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()

        model.train()

        for batch_x, batch_y in train_loader:
            x, y = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_batch_loss = loss(pred, y)
            train_batch_loss.backward()
            # 更新模型
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()
            train_loss += train_batch_loss.cpu().item()
        plt_train_loss.append(train_loss / train_loader.__len__())

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, y = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_batch_loss = loss(pred, y)
                val_loss += val_batch_loss.cpu().item()
        plt_val_loss.append(val_loss / val_loader.__len__())
        if val_loss < min_val_loss:
            torch.save(model, save_path)
            min_val_loss = val_loss
        print("[%03d/%03d] %2.2f sec(s) TrainLoss:%.6f | ValLoss:%0.6f" % (
            epoch, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1]))

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss图")
    plt.legend(["train","val"])
    plt.show()

def evaluate(save_path, test_loader, device, res_path):
    model = torch.load(save_path).to(device)
    res = []
    with torch.no_grad():
        for x in test_loader:
            pred = model(x.to(device))
            res.append(pred.cpu().item())
    print(res)

    with open(res_path, "w",newline="") as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(["id","tested_positive"])
        for i, value in enumerate(res):
            csvWriter.writerow([str(i), str(value)])
    print("文件已经保存到"+res_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
config = {
    "lr": 0.001,
    "epochs": 20,
    "momentum": 0.9,  # 表示动量
    "save_path": "model_save1/best_model.pth",
    "res_path":"pred.csv"
}
model = MyModel(inDim=93).to(device)
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),
                      lr=config["lr"],
                      momentum=config["momentum"])
# train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])
evaluate(config["save_path"], test_loader, device, config["res_path"])
