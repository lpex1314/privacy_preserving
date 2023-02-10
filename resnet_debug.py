import torch.nn as nn
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from replace import register_op
from op_enc2.CrossEntropy import CrossEntropy
from preprocess import getdll2, preprocess
from ctypes import *
from hooks.hook import register_hook, register_for_hook
from storage import layer_list
forward, backward = getdll2()

#残差块
class ResidualBlock(nn.Module):
    def __init__(self,channel):
        super(ResidualBlock, self).__init__()
        self.channel=channel
        self.conv1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(channel)
    def forward(self,x):
        out=self.conv1(x)
        out=self.BN(out)
        out=self.relu(out)
        out=self.conv2(out)
        out+=x
        out=self.relu(out)
        return out

 
#残差网络
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5,bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)  
        self.reslayer1=ResidualBlock(32)
        self.reslayer2=ResidualBlock(16)
        self.fc=nn.Linear(256,10, bias=False)              #这里的输入256是因为16*4*4=256

 
    def forward(self,x):
        
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)
        print(type(out))
        print(out.shape)
        out=self.reslayer1(out)
        print(out.shape)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.maxpool(out)
        print(out.shape)
        out=self.reslayer2(out)
        print(out.shape)
        out=out.view(out.size(0),-1)
        print(out.shape)
        out=self.fc(out)
        print(out.shape)
        return out


 
device = 'cpu' if torch.cuda.is_available() else 'cpu'
print('device =', device)
Epoch=3
Batch_Size=50
LR=0.0001
classes=10
net=ResNet().to(device)
net = register_op(net)
register_for_hook(net)
#训练集
trainData=torchvision.datasets.MNIST(
    root="/home/lpx/codes/hook/data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)
 
train_loader=DataLoader(dataset=trainData,batch_size=Batch_Size,shuffle=True)
test_data=torchvision.datasets.MNIST(root="home/lpx/codes/hook/data",train=False,download=True)
 
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.float)[:5000]/255. # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:5000]



#关于训练
def Train(Res):
    # 损失函数,以及优化器
    loss_func = CrossEntropy()
    optimizer = torch.optim.Adam(Res.parameters(), lr=LR)
    for epoch in range(Epoch):
        for step,(b_x,b_y)in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            N, C, H, W = b_x.shape
            res = torch.rand([N * 10, C, H, W])
            shape_x, x, len_x, double_array_x = preprocess(b_x)
            shape_res, res, len_res, double_array_res = preprocess(res)
            forward.ecall_encrypt.argtypes = (double_array_x, c_int, double_array_res)
            forward.ecall_encrypt(x, c_int(len_x), res)
            input = np.frombuffer(res, dtype=np.double)
            input = torch.tensor(input, dtype=torch.float)
            input = input.reshape(*shape_res)  # [500, 1, 28, 28]
            input = input.to(device)
            output=Res(input)
            target_one_hot = torch.zeros(Batch_Size, 10).scatter(1, b_y.view(Batch_Size,1), 1) 
            loss=loss_func(output, target_one_hot, classes)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # clear the temporary storage list of inputs and layers after each forward and backward round process
            layer_list.inputs.clear()
            layer_list.layers.clear()
            layer_list.n = 0
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            # if(step%50==0):
            #     test_output=Res(test_x)
            #     pred_y = torch.max(test_output, 1)[1].data.numpy()
            #     accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.6f' % accuracy)
 
    # torch.save(Res, 'res_minist.pkl')
    print('res finish training')

Train(net)
