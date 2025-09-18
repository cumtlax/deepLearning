
import csv
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



def processdate(date):
    date_num = (int(date[:4]) - 2014)*12 + (int(date[4:6])-5)
    return date_num


class houseDataset(Dataset):
    def __init__(self, path, mode="train", feature_dim=5):
        with open(path,'r') as f:
            csv_data = list(csv.reader(f))
            for index in range(1, len(csv_data)):
                csv_data[index][1] = processdate(csv_data[index][1])
                csv_data[index][2] = str(eval(csv_data[index][2]))
            x = np.delete(np.array(csv_data)[1:].astype(float), [0, 2, 16], axis=1)
            y = np.array(csv_data)[1:, 2].astype(float)/10**6
            self.x = torch.tensor(x)
            self.y = torch.tensor(y)
            self.x = (self.x - self.x.mean(dim=0,keepdim=True)) / self.x.std(dim=0,keepdim=True)
            print('Finished reading the {} set of house Dataset ({} samples found)'
                  .format(mode, len(self.x)))

    def __getitem__(self, item):
            return self.x[item].float(), self.y[item]
    def __len__(self):
        return len(self.x)


class myNet(nn.Module):
    def __init__(self,inDim):
        super(myNet,self).__init__()
        self.fc1 = nn.Linear(inDim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        # self.fc3 = nn.Linear(256,256)
        # self.fc4 = nn.Linear(inDim,1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        #
        # x = self.fc3(x)
        # x = self.relu(x)
        #
        # x = self.fc4(x)
        # x = self.relu(x)

        if len(x.size()) > 1:
            return x.squeeze(1)
        else:
            return x




def train_val(model, trainloader, valloader,optimizer, loss, epoch, device, save_):

    # trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
    # valloader = DataLoader(valset,batch_size=batch,shuffle=True)
    model = model.to(device)
    plt_train_loss = []
    plt_val_loss = []
    val_rel = []
    min_val_loss = 100000
    for i in range(epoch):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        for data in trainloader:
            optimizer.zero_grad()
            x, target = data[0].to(device), data[1].to(torch.float32).to(device)
            pred = model(x)
            bat_loss = loss(pred, target, model)
            bat_loss.backward()
            optimizer.step()
            train_loss += bat_loss.detach().cpu().item()

        plt_train_loss. append(train_loss/trainloader.dataset.__len__())

        model.eval()
        with torch.no_grad():
            for data in valloader:
                val_x , val_target = data[0].to(device), data[1].to(device)
                val_pred = model(val_x)
                val_bat_loss = loss(val_pred, val_target, model)
                val_loss += val_bat_loss.detach().cpu().item()
                val_rel.append(val_pred)
        if val_loss < min_val_loss:
            torch.save(model, save_)

        plt_val_loss. append(val_loss/valloader.dataset.__len__())

        print('[%03d/%03d] %2.2f sec(s) TrainLoss : %3.6f | valLoss: %3.6f' % \
              (i, epoch, time.time()-start_time, plt_train_loss[-1], plt_val_loss[-1])
              )

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('loss')
    plt.legend(['train', 'val'])
    plt.show()



from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_path = 'kc_house_data.csv'
house_data = pd.read_csv(train_path)

print(house_data.head())

house_data.info()

print(house_data.describe())

house_dataset = houseDataset(train_path)

train_set, val_set = random_split(house_dataset,[int(len(house_dataset)*0.8),len(house_dataset)- int(len(house_dataset)*0.8)])
print(train_set.indices)



def mseLoss(pred, target, model):
    loss = nn.MSELoss(reduction='mean')
    ''' Calculate loss '''
    regularization_loss = 0
    for param in model.parameters():
        # TODO: you may implement L1/L2 regularization here
        # 使用L2正则项
        # regularization_loss += torch.sum(abs(param))
        regularization_loss += torch.sum(param ** 2)
    return loss(pred, target) + 0.00075 * regularization_loss

loss =  mseLoss

config = {
    'n_epochs': 50,                # maximum number of epochs
    'batch_size': 25,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0001,                 # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'model_save/model.pth',  # your model will be saved here
}

model = myNet(18).to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.03,momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.03,
                             weight_decay=0.001)     #换一个优化器


trainloader = DataLoader(train_set,batch_size=config['batch_size'], shuffle=True)
valloader = DataLoader(val_set,batch_size=config['batch_size'], shuffle=True)

train_val(model, trainloader,valloader,optimizer, loss, config['n_epochs'], device,save_=config['save_path'])




















