# 这个代码文件用来测试和验证应用乘法分配律复合求导的思想，以及验证加密模型和原生模型各中间值diff
import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torch import nn
from ctypes import *
from replace import register_op
from hooks.hook import register_for_hook
from storage import layer_list
from preprocess import getdll, preprocess
from globals import global_param
from op.CrossEntropy import CrossEntropy
import numpy as np
import matplotlib.pyplot as plt
Ne = global_param.num_segmentation
forward, backward = getdll()

device = 'cpu' if torch.cuda.is_available() else 'cpu'
print('device =', device)
Epoch=1
Batch_Size=50
LR=0.0001
num_classes=10
step_lim = 10
n1_w1_list = []
n1_w2_list = []
n2_w1_list = []
n2_w2_list = []
w11 : Tensor
w11 = torch.zeros([1, 28, 28])
w12 = torch.zeros([1, 28, 28])
w21 = torch.zeros([1, 28, 28])
w22 = torch.zeros([1, 28, 28])
trainData=torchvision.datasets.MNIST(
    root="/home/lpx/codes/hook/data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)
train_loader=DataLoader(dataset=trainData,batch_size=Batch_Size,shuffle=False,drop_last=True)
test_data=torchvision.datasets.MNIST(root="home/lpx/codes/hook/data",train=False,download=True)
# 1, 28, 28
print(len(train_loader))
class Net(nn.Module):
    def __init__(self, out_channels) -> None:
        super().__init__()
        self.w1 = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            nn.Dropout(p=0.2),
        ) 
        self.fc = nn.Linear(in_features=28*28*out_channels, out_features=num_classes, bias=False)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.w1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.relu(x)
        # x = self.softmax(x)
        return x
    
    def initialize_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1)
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                    #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.fill_(1)
                # m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

# def register_one(model):
#     for _, i in model.named_modules():
#         if not isinstance(i, nn.Sequential):
            # if isinstance(i, nn.Conv2d) or isinstance(i, nn.Linear):  # 如果是线性层
                # print(i)
            # i.register_forward_hook(hook_for)
            # i.register_backward_hook(hook_back) 
# def hook_for(module, input, output):
    # print(module, 'output max:', output.max().item())
    
def hook_back(module, grad_in, grad_out):
    dy = grad_out[0]
    # if len(grad_in) > 2:
    #     if isinstance(module, nn.Linear):
    #         db, dx, dw = grad_in
    #     else:
    #         dx, dw, db = grad_in
    # else:
    #     dx, dw = grad_in
    # # print(dx, dw)
    print(module, 'grad_out max ', dy.max().item())
        
net = Net(out_channels=5)
# unencrypted version
print('-----unencrypted model-----')
# register_one(net)
net.initialize_params()
# for name, param in net_.named_parameters():
    # print(param.data.flatten()[:10])
    #tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    #tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
def Train_1(model):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(Epoch):
        for step,(b_x,b_y)in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            loss=loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: ', epoch, 'step: ', step, '| train loss: %.4f' % loss.data.numpy())
            
            for name, param in model.named_parameters():
                if name == 'w1.0.weight':
                    print('conv2d weight: ', param.data.flatten()[:5])
                    print('conv2d weight grad: ', param.grad.data.flatten()[:5])
                    n1_w1_list.append(param.grad.data.flatten()) 
                if name == 'fc.weight':
                    print('linear weight: ', param.data.flatten()[:5])
                    print('linear weight grad: ', param.grad.data.flatten()[:5])
                    n1_w2_list.append(param.grad.data.flatten()) 
            if step == step_lim:    
                break
                # tensor([1.0459, 1.0507, 1.0470, 1.0548, 1.0582, 1.0536, 1.0436, 1.0491, 1.0459, 1.0459])
    print('res finish training')

Train_1(net)

# encrypted version
print('-----encrypted model-----')
net_ = Net(out_channels=5)
net_ = register_op(net_)
register_for_hook(net_)
net_.initialize_params()
# for name, param in net_.named_parameters():
    # print(param.data.flatten()[:10])
    #tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    #tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
def Train_2(model):
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
            # 下面这段代码用来测试，backward_hook中返回的grad_input确实经过了modified，且其中的
            # dw也成功赋值给了module.weight.grad
            # print('backward is over')
            # for name, param in model.named_parameters():
            #     if name == 'w1.0.weight':
            #         print(param.grad.flatten()[:10])
            #     if name == 'fc.weight':
            #         print(param.grad.flatten()[:10])
            
            # clear the temporary storage list of inputs and layers after each forward and backward round process
            layer_list.inputs.clear()
            layer_list.layers.clear()
            layer_list.n = 0
            print('Epoch: ', epoch, 'step: ', step, '| train loss: %.4f' % loss.data.numpy())
            
            for name, param in model.named_parameters():
                if name == 'w1.0.weight':
                    print('conv2d weight: ',param.data.flatten()[:5])
                    print('conv2d weight grad: ', param.grad.data.flatten()[:5])
                    # w21 = param.data.flatten()
                    n2_w1_list.append(param.grad.data.flatten()) 
                    # print(param.data.flatten().shape)
                if name == 'fc.weight':
                    print('linear weight: ',param.data.flatten()[:5])
                    print('linear weight grad: ', param.grad.data.flatten()[:5])
                    # w22 = param.data.flatten()
                    n2_w2_list.append(param.grad.data.flatten()) 
            if step == step_lim:
                break
            
        # for name, param in model.named_parameters():
        #     if name == 'w1.0.weight' or name == 'fc.weight':
        #         w2 = param.data.flatten()
        #         print(param.data.flatten()[:10])
            # if(step%50==0):
            #     test_output=Res(test_x)
            #     pred_y = torch.max(test_output, 1)[1].data.numpy()
            #     accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.6f' % accuracy)
 
    # torch.save(Res, 'res_minist.pkl')
    print('res finish training')

Train_2(net_)
# print(w11.max())
# print(w12.max())
# print(w21.max())
# print(w22.max())
# print(((w11 - w21) / w11).max())
# print(((w12 - w22) / w12).max())
print('-----Relative error-----')
a1= torch.tensor([item.detach().numpy() for item in n1_w1_list])
a2= torch.tensor([item.detach().numpy() for item in n1_w2_list])
b1= torch.tensor([item.detach().numpy() for item in n2_w1_list])
b2= torch.tensor([item.detach().numpy() for item in n2_w2_list])
# print(a1.max())
print('conv2d grad relative error:', ((a1 - b1) / a1).max() )
print('matmul grad relative error:', ((a2 - b2) / b2).max() )
# LR=0.01 step_lim = 0 可以发现一个step之后的loss、weight几乎相同，说明正反向计算过程相同