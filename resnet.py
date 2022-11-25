import torch.nn as nn
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from replace import register_op
from op_enc2.CrossEntropy import CrossEntropy
from preprocess import getdll2, preprocess
from ctypes import *
from hooks.hook_forward import register_hook
forward, backward = getdll2()

#残差块
class ResidualBlock(nn.Module):
    def __init__(self,channel):
        super(ResidualBlock, self).__init__()
        self.channel=channel
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=channel,
                      out_channels=channel,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False),
            # nn.BatchNorm2d(channel)
        )
        self.relu = nn.ReLU()
 
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out+=x
        out=self.relu(out)
        return out

 
#残差网络
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,bias=False), #(1,28,28)
            nn.BatchNorm2d(32),                                     #(32,24,24)
            nn.ReLU(),
            nn.MaxPool2d(2)                                         #(32,12,12)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5,bias=False), #(16,8,8)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)                                           #(16,4,4)
        )
        self.reslayer1=ResidualBlock(32)
        self.reslayer2=ResidualBlock(16)
        self.fc=nn.Linear(256,10, bias=False)              #这里的输入256是因为16*4*4=256
        
 
    def forward(self,x):
        out=self.conv1(x)
        out=self.reslayer1(out)
        out=self.conv2(out)
        out=self.reslayer2(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out
 
# criterion = CrossEntropy()
# loss_func = nn.CrossEntropyLoss()
# net1 = ResNet()
# net2 = ResNet()
# register_op(net1)
# register_hook(net2)
# # for name, module in net1.named_modules():
# #     if type(module) != nn.Sequential:
# #         print(module)
# origin_input = torch.rand([4, 1, 28, 28])
# res = torch.rand([40, 1, 28, 28])
# shape_x, x, len_x, double_array_x = preprocess(origin_input)
# shape_res, res, len_res, double_array_res = preprocess(res)
# forward.ecall_encrypt.argtypes = (double_array_x, c_int, double_array_res)
# forward.ecall_encrypt(x, c_int(len_x), res)
# input = np.frombuffer(res, dtype=np.double)
# input = torch.tensor(input, dtype=torch.float)
# input = input.reshape(*shape_res)  # [40, 1, 28, 28]
# origin_input=origin_input.view(origin_input.size(0),-1)
# input=input.view(input.size(0),-1)
# m = nn.Linear(784, 20, bias=False)   
# test_1 = m(input)
# test_2 = m(origin_input)
# shape_1, encrypted, len_1, double_array_1 = preprocess(test_1)
# shape_new, _, len_new, _ = preprocess(test_2)
# result = torch.rand(*shape_new)
# shape_re, ans, len_re, double_array_re = preprocess(result)
# forward.ecall_decrypt.argtypes = (double_array_1, c_int, double_array_re)
# forward.ecall_decrypt(encrypted, c_int(len_new), ans)
# decrypted = np.frombuffer(ans, np.double)
# decrypted = torch.tensor(decrypted, dtype=torch.float)
# decrypted = decrypted.reshape(*shape_new)
# print((decrypted-test_2).max())

# labels = [1, 3, 7, 2]
# labels = torch.tensor(labels)
# target_one_hot = torch.zeros(4, 10).scatter(1, labels.view(4,1),1) 
# print(origin_input.mean())
# print(input.mean())
# print('**my_model**')
# y_pre1 = net1(input)
# loss_myModel = criterion(y_pre1, target_one_hot, classes=10)
# print('**nn**')
# y_pre2 = net2(origin_input)
# loss_nn = loss_func(net2(origin_input), labels)
# print(loss_myModel)
# print(loss_nn)

 

Epoch=3
Batch_Size=50
LR=0.0001
classes=10
net=ResNet()
register_op(net)
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
            N, C, H, W = b_x.shape
            res = torch.rand([N * 10, C, H, W])
            shape_x, x, len_x, double_array_x = preprocess(b_x)
            shape_res, res, len_res, double_array_res = preprocess(res)
            forward.ecall_encrypt.argtypes = (double_array_x, c_int, double_array_res)
            forward.ecall_encrypt(x, c_int(len_x), res)
            input = np.frombuffer(res, dtype=np.double)
            input = torch.tensor(input, dtype=torch.float)
            input = input.reshape(*shape_res)  # [40, 1, 28, 28]
            output=Res(input)
            target_one_hot = torch.zeros(Batch_Size, 10).scatter(1, b_y.view(Batch_Size,1),1) 
            loss=loss_func(output, target_one_hot, classes)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            if(step%50==0):
                test_output=Res(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.6f' % accuracy)
 
    # torch.save(Res, 'res_minist.pkl')
    print('res finish training')

Train(net)
