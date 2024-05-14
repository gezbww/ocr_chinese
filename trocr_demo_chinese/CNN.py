import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from glob import glob
batch_size = 512
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 1. 定义自定义的Dataset类
class MyCustomDataset(Dataset):
    def __init__(self, data, labels,transformer):
        self.data = data
        self.labels = labels
        self.transformer=transformer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 根据索引获取单个样本
        img = self.data[index]
        image = Image.open(img)#.convert('RGB')
        label = self.labels['label'][index]
        if self.transformer:
            image = self.transformer(image)
        #label = self.labels[index]
        return image, label
df_train=pd.read_fwf("")
df_train.columns=['label']
df_test=pd.read_fwf("")
df_test.columns=['label']
train_path=glob('')
test_path=glob(')
'''
#1、导入数据，并且查看数据，MINIST数据集包含train和test数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)
'''
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((16, 16)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_dataset=MyCustomDataset(train_path,df_train,transform)
test_dataset=MyCustomDataset(test_path,df_test,transform)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=False,pin_memory=True)
x, y = next(iter(train_loader))
print(x, y)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
#2、定义网络，根据卷积层--池化层--卷积层--池化层--全连接层连接方式定义
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 4 * 4, 12000),
            nn.ReLU(inplace=True),
            nn.Linear(12000, 8400),
            nn.ReLU(inplace=True),
            nn.Linear(8400, 3076)
        )
    def forward(self, x):
        # param x:[512,1,28,28]

        batchsz = x.size(0)
        # [512,1,28,28]->[512,16,4,4]
        x = self.conv_unit(x)

        # 动态计算全连接层的输入特征数
        #fc_input_features = x.numel() // batchsz
        #x = x.view(batchsz, fc_input_features)
        # [512,1,28,28]->[512,16*4*4]
        x = x.view(batchsz, 16 * 4 * 4)
        # [512,16*4*4]->[512,10]
        logits = self.fc_unit(x)
        # #[512,10]
        # pred = F.softmax(logits,dim=1)
        # loss = self.criteon(logits,y)
        return logits


#3、初始化网络，并且打印出模型基本结构
model=Cnn()
criteon = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
print(model)
model.to(device)
#4、训练网络，并保存训练中的loss以及acc等
train_loss = []
Acc = []
for epoch in range(30):
    model.train()
    running_loss = 0
    for batchidx, (x, label) in enumerate(train_loader):
        x = x.to(device)
        #label = label.to(device)

        logits = model(x)
        loss = criteon(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss.append(running_loss / len(train_loader))
    print('epoch:', epoch + 1, 'loss:', loss.item())

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in test_loader:

            logits = model(x)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
        acc = total_correct / total_num
        Acc.append(total_correct / total_num)
        print('test acc:', acc)

#5、可视化train的loss以及test的acc
plt.plot(train_loss, label='Loss')
plt.plot(Acc, label='Acc')












