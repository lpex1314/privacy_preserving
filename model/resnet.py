import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
# from op_nn import ReLU
# from op_nn import MaxPool2d
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
                      padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(channel)
        )
        self.conv = nn.Conv2d(channel,channel,kernel_size=3,stride=1)
        self.relu = nn.ReLU()
        
 
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out+=x
        out=nn.ReLU(out)
        return out

 
#残差网络
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5), #(1,28,28)
            nn.BatchNorm2d(32),                                     #(32,24,24)
            nn.ReLU(),
            nn.MaxPool2d(2)                                         #(32,12,12)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5), #(16,8,8)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)                                           #(16,4,4)
        )
        self.reslayer1=ResidualBlock(32)
        self.reslayer2=ResidualBlock(16)
        self.fc=nn.Linear(256,10)              #这里的输入256是因为16*4*4=256
        
 
    def forward(self,x):
        out=self.conv1(x)
        out=self.reslayer1(out)
        out=self.conv2(out)
        out=self.reslayer2(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return  out
 
net = ResNet()
for name, module in net.named_modules():
    print(name)
    # print(module)
 
 
# Epoch=3
# Batch_Size=50
# LR=0.01
 
# #训练集
# trainData=torchvision.datasets.MNIST(
#     root="/home/lpx/codes/hook/data",
#     train=True,
#     transform=torchvision.transforms.ToTensor(),
#     download=True)
 
# train_loader=DataLoader(dataset=trainData,batch_size=Batch_Size,shuffle=True)
# test_data=torchvision.datasets.MNIST(root="home/lpx/codes/hook/data",train=False,download=True)
 
# test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.float)[:5000]/255. # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.targets[:5000]


 
# #关于训练
# def Train(Res):
#     # 损失函数,以及优化器
#     loss_func = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(Res.parameters(), lr=LR)
#     for epoch in range(Epoch):
#         for step,(b_x,b_y)in enumerate(train_loader):
#             output=Res(b_x)
#             loss=loss_func(output,b_y)
 
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
 
#             if(step%50==0):
#                 test_output=Res(test_x)
#                 pred_y = torch.max(test_output, 1)[1].data.numpy()
#                 accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#                 print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.6f' % accuracy)
 
#     # torch.save(Res, 'res_minist.pkl')
#     print('res finish training')


