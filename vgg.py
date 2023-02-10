import torch
import torch.nn as nn
from replace import register_op
from hooks.hook import register_for_hook
from storage import layer_list
import torchvision
from preprocess import getdll, preprocess
from torch.utils.data import DataLoader
from op.CrossEntropy import CrossEntropy
from ctypes import *
import numpy as np
from globals import global_param
Ne = global_param.num_segmentation
forward, backward = getdll()

VGG_types = {
"VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, 512, 512],
"VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, 512, 512],
"VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,512,512,512],
"VGG19": [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,
          512,512,512,512]}


VGGType = "VGG16"
class VGGnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGGnet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[VGGType])
        self.fcs = nn.Sequential(
        nn.Linear(512 * 3 * 3, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print('shape' + str(x.shape))
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                ),
                nn.BatchNorm2d(x),
                nn.ReLU()
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

# device = "cpu" if torch.cuda.is_available() else "cpu"
# model = VGGnet(in_channels=3, num_classes=500).to(device)
# register_op(model)
# print(model)
# x = torch.randn(1, 3, 224, 224).to(device)
# print(model(x).shape)

device = 'cpu' if torch.cuda.is_available() else 'cpu'
print('device =', device)
Epoch=3
Batch_Size=50
LR=0.0001
num_classes=10
net=VGGnet(in_channels=1, num_classes=num_classes).to(device)
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

# for name, param in net.named_parameters():
#     if param.requires_grad:
#         print(name)

#关于训练
def Train(model):
    # 损失函数,以及优化器
    loss_func = CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(Epoch):
        for step,(b_x,b_y)in enumerate(train_loader):
            # print(b_x.shape)
            N, C, H, W = b_x.shape
            res = torch.rand([N * Ne, C, H, W])
            shape_x, x, len_x, double_array_x = preprocess(b_x)
            shape_res, res_c, len_res, double_array_res = preprocess(res)
            forward.ecall_encrypt.argtypes = (double_array_x, c_int, double_array_res)
            forward.ecall_encrypt(x, c_int(len_x), res_c)
            input = np.frombuffer(res_c, dtype=np.double)
            input = torch.tensor(input, dtype=torch.float)
            input = input.reshape(*shape_res)  # [500, 1, 28, 28]
            input = input.to(device)
            
            output = model(input)
            target_one_hot = torch.zeros(Batch_Size, num_classes).scatter(1, b_y.view(Batch_Size,1), 1) 
            loss=loss_func(output, target_one_hot, num_classes)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # clear the temporary storage list of inputs and layers after each forward and backward round process
            layer_list.inputs.clear()
            layer_list.layers.clear()
            layer_list.n = 0
            for name, param in model.named_parameters():
                if name == 'conv_layers.0.weight' or name == 'fcs.6.weight':
                    print(param.data.flatten()[:10])
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            # if(step%50==0):
            #     test_output=Res(test_x)
            #     pred_y = torch.max(test_output, 1)[1].data.numpy()
            #     accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.6f' % accuracy)
 
    # torch.save(Res, 'res_minist.pkl')
    print('res finish training')

Train(net)