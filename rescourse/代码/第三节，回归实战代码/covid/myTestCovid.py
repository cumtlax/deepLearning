import csv
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class CovidDataset(Dataset):
    def __init__(self, file_path, mode):
        with open(file_path, "r") as f:
            origin_data = list(csv.reader(f))
            temp_data = np.array(origin_data)
            csv_data = temp_data[1:, 1:].astype(float)
            if mode == "train":
                indices = [i for i in range(len(csv_data)) if i % 5 != 0]
                # 取出x
                x = torch.tensor(csv_data[indices, :-1])
                # 取出y
                self.y = torch.tensor(csv_data[indices, -1])
            elif mode == "val":
                indices = [i for i in range(len(csv_data)) if i % 5 == 0]
                x = torch.tensor(csv_data[indices, :-1])
                self.y = torch.tensor(csv_data[indices, -1])
            elif mode == "test":
                indices = [i for i in range(len(csv_data))]
                x = torch.tensor(csv_data[indices,:-1])
            # 权重归一化
            self.x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
            self.mode = mode

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.mode != "test":
            return self.x[idx].float(), self.y[idx].float()
        else:
            return self.x[idx].float()

train_file = "covid.train.csv"
test_file = "covid.test.csv"
train_dataset = CovidDataset(train_file, "train")
val_dataset = CovidDataset(train_file, "val")
test_dataset = CovidDataset(train_file, "test")

batch_size = 16
train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size = batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size = 1,shuffle=False)

class myModel(nn.Module):
    def __init__(self, inDim):
        super(myModel, self).__init__()
        self.fc1 = nn.Linear(inDim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.sigmoid1 = nn.Sigmoid()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid1(x)
        x = self.fc3(x)
        if len(x.size()) > 1:
            return x.squeeze(1)
        return x

def train_val(model,train_loader,val_loader,device,epochs,loss,save_path):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    model = model.to(device)

    plt_train_loss = []
    plt_val_loss = []

    min_val_loss = 99999999

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()

        model.train()

        for batch_x, batch_y in train_loader:
            x, y = batch_x.to(device), batch_y.to(device)
            pred_y = model(x)
            train_batch_loss = loss(pred_y, y)
            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += train_batch_loss.cpu().item()
        plt_train_loss.append(train_loss/train_loader.__len__())

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, y = batch_x.to(device), batch_y.to(device)
                pred_y = model(x)
                val_batch_loss = loss(pred_y, y)
                val_loss += val_batch_loss.cpu().item()
        plt_val_loss.append(val_loss/val_loader.__len__())
        if val_loss < min_val_loss:
            val_loss = min_val_loss
            torch.save(model,save_path)
        print("[%03d/%03d] %2.2f sec(s) TrainLoss:%.6f | ValLoss:%0.6f" % (
            epoch, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1]))

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss")
    plt.legend(["train_loss","val_loss"])
    plt.show()

def evaluate(save_path, test_loader, device, res_path):
    model = torch.load(save_path).to(device)
    res=[]
    with torch.no_grad():
        for x in test_loader:
            pred_y = model(x.to(device))
            res.append(pred_y.cpu().item())
    print(res)

    with open(res_path,"w", newline="") as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(["id","tested_positive"])
        for i, value in enumerate(res):
            csvWriter.writerow([str(i), str(value)])
    print("文件已经保存到" + res_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
config = {
    "lr":0.001,
    "epochs":20,
    "momentum":0.9,
    "save_path":"model_save1/best_model1.path",
    "res_path":"pred.csv"
}

model = myModel(inDim=93).to(device)
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),
                      lr=config["lr"],
                      momentum=config["momentum"])

# train_val(model,train_loader,val_loader,device,config["epochs"],loss,config["save_path"])
evaluate(config["save_path"],test_loader,device,config["res_path"])