"""加载依赖库"""
#first MnistProgram second handin
import time

import  torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
time0=time.time()
'''定义超参数'''
batch_size=3000#每批处理的数据
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")#用什么数据训练
epochs=100#训练多少轮
learning_rate = 0.05

'''构建pipeline 转换图片'''
pipeline=transforms.Compose([
    transforms.ToTensor(),#将图片转换为tensor
    transforms.Normalize((0.1307,), (0.3081,))#正则化,防止过拟合问题
])
'''下载数据集'''

trainSet=datasets.MNIST('data',train=True, download=True, transform=pipeline)
testSet=datasets.MNIST("data", train=False, download=True, transform=pipeline)#测试集改为false


trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)#shuffle打乱顺序，提高模型精度
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True)

#显示MNIST中图片

#构建网络模型

class MyModel(nn.Module):#搭建网络模型
    def __init__(self):
        super().__init__()
        '''卷积层，共两层'''
        self.conv1 = nn.Conv2d(1,10,5)#(inputChannel,outputChannel,Kernel)
        self.conv2 = nn.Conv2d(10,20,3)#(InputChannel,OutputChannel,Kernel)
        '''全连接层'''
        self.fc1 = nn.Linear(20*10*10,500)#(InputChannel,OutputChannel)
        self.fc2 = nn.Linear(500, 10)#与输出结果链接(InputChannel,OutputChannel)
    def forward(self, x):
        input_size=x.size(0)#输入给模型的数据格式是  batchSize*1*28*28，0取batchSize
        x=self.conv1(x)#输入的x格式为batch*1*24*24  变为24的原因是因为 Kernel为5*5，边缘两个像素被丢掉了
        #x=F.relu(x)#激活函数，保持shape不变
        x=F.max_pool2d(x,2,2)#对图片降采样 (data，KernelX,KernelY),输入：batchSize*10*24*24，输出：batchSize*10*12*12

        x=self.conv2(x)#输入：batchSize*10*20*20，输出：batchSize*20*10*10
        X=F.relu(x)

        x=x.view(input_size,-1)#拉平(inputsize,size)-1表示自动计算维度

        x=self.fc1(x)#输入 batchSize*2K 输出 batchSize*500
        x=F.relu(x)#激活保持shape不变

        x=self.fc2(x)#输入batchSize*500 输出 batchSize*10

        output = F.log_softmax(x,dim=1)#计算分类后每个数字的概率

        return output#返回概率值

model=MyModel().to(device)

optimizer=optim.Adam(model.parameters())

def trainModel(model,device,train_loader, optimizer, epoch):#模型训练
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #数据部署上device
        data, target = data.to(device), target.to(device)
        #梯度初始化
        optimizer.zero_grad()
        #训练后的结果
        output=model(data)
        #计算损失
        loss=F.cross_entropy(output, target)#交叉熵损失，针对多分类任务(预测值，真实值)
        #反向传播
        loss.backward()
        #参数优化
        optimizer.step()
        if batch_idx % 30000==0:
            print('Train Epoch : {} \t loss= {:.6f}'.format(epoch,loss.item()))

def test(model, device, test_loader):
    #模型验证
    model.eval()
    #正确率,损失
    correct = 0.0
    testLoss = 0.0
    #进行测试
    with torch.no_grad():#测试不计算梯度,不进行反向传播
        for data, target in test_loader:
            #部署到device上
            data, target = data.to(device), target.to(device)
            #测试
            output=model(data)
            testLoss+=F.cross_entropy(output, target).item()
            #找出最大
            pred = output.argmax(dim=1)
            #pred=output.max(dim=1,keepdim=True)[1]//找到的是值+索引
            #pred=torch.max(output,dim=1)
            #计算正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        testLoss /= len(test_loader.dataset)#归一化的错误概率
        print('Test Average Error ={:.4f} ,Accuracy = {:.4f}\n'.format(testLoss,100*correct/len(test_loader.dataset)))

#调用
for epoch in range(1,epochs + 1):
   trainModel(model,device,trainLoader,optimizer,epoch)
   if epoch==50 or epoch ==100:
    test(model, device, testLoader)
time1=time.time()
print(time1-time0)
#approximately 99%