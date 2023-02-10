# 此python文件用来验证，使用torch.autograd.grad输入weight、y、grad_output来求dloss/dw的方法是否正确
# 验证成果：正确！
import torch
import torchvision
import torch.nn as nn
from typing import Union
from torch import Tensor
from torch.autograd import Variable, variable
from torch.utils.data import DataLoader
VGG_types = {
"VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
"VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
"VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,512,512,512],
"VGG19": [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,
          "M",512,512,512,512,"M",],}

# this class aims to record down all the inputs, outputs and layers of a running model
class trace_layers():
    def __init__(self) -> None:
        self.inputs = []
        self.outputs = []
        self.layers = []
        self.n = 0
        pass
    
    def update(self, model, input, output):
        self.n += 1
        self.layers.append(model)
        self.inputs.append(input)
        self.outputs.append(output)

        
    def get_cur(self):
        layer = self.layers[self.n - 1]
        x = self.inputs[self.n - 1]
        y = self.outputs[self.n - 1]
        self.n -= 1
        return layer, x, y

layer_list = trace_layers()

# build model
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
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                ),
                nn.BatchNorm2d(x),
                nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

# determine whether it is a linear layer
def isLinear(feature):
    if isinstance(feature, nn.Linear) or isinstance(feature, nn.Conv2d):
        return True
    else:
        return False
    
def register_for_hook(model):
    for _, i in model.named_modules():
        if not isinstance(i, nn.Sequential):
            if isLinear(i):  
                i.register_forward_hook(forward_hook)
                i.register_backward_hook(backward_hook) 

def compute_grad(input: Tensor, layer, grad_out : Tensor):
    x = input.clone().detach()
    x.requires_grad = True
    # print(x.shape)
    # [50, 512] 50 is batch_size
    grad_out = grad_out.clone().detach()
    if isinstance(layer, nn.Linear):
        func = nn.Linear(in_features=layer.in_features, out_features=layer.out_features, bias=True)
    else:
        layer: nn.Conv2d
        func = nn.Conv2d(in_channels=layer.in_channels, out_channels=layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride\
            ,padding=layer.padding)
    func.load_state_dict(layer.state_dict())
    y = func(x)
    grad_w = torch.autograd.grad(y, func.weight, grad_out)
    return grad_w

def self_grad(input: Tensor, layer, gout : Tensor):
    print(gout.grad_fn)
    grad_w = torch.autograd.grad(gout, layer.weight, grad_outputs=torch.ones_like(gout))
    return grad_w

def forward_hook(module, input, output):
    layer_list.update(module, input, output)
    return None
    
def backward_hook(module, gin, gout):
    layer, input, output = layer_list.get_cur()
    input = input[0]
    dy = gout[0]
    if isinstance(module, nn.Linear):
        db, dx, dw = gin
    elif isinstance(module, nn.Conv2d):
        dx, dw, db = gin
    cast = dy / output
    # call my custom compute_grad function
    # print(len(gin))
    # 3
    # print(input.shape, module.bias.shape, module.weight.shape)
    # torch.Size([50, 512]) torch.Size([10]) torch.Size([10, 512])
    # print(dx.shape, db.shape, dw.shape)
    # 第一层dx为None
    # torch.Size([50, 512]) torch.Size([10]) torch.Size([512, 10])
    # the dw.shape is the transpose of module.weight.shape, the pytorch will transpose the
    # shape of weight during actual computation
    with torch.enable_grad():
        self_dw = compute_grad(input, module, dy)[0]
    # print(self_dw.shape)
    # print(dw.shape)
    print(((self_dw.T - dw)))
    # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
    #     [0., 0., 0.,  ..., 0., 0., 0.],
    #     [0., 0., 0.,  ..., 0., 0., 0.],
    #     ...,
    #     [0., 0., 0.,  ..., 0., 0., 0.],
    #     [0., 0., 0.,  ..., 0., 0., 0.],
    #     [0., 0., 0.,  ..., 0., 0., 0.]])
    assert self_dw.T == dw
    if isinstance(module, nn.Linear):
        new_gin = tuple([db, dx, self_dw.T])
    elif isinstance(module, nn.Conv2d):
        new_gin = tuple([dx, self_dw, db])
    return new_gin

# initializing
device = 'cpu' if torch.cuda.is_available() else 'cpu'
print('device =', device)
Epoch=3
Batch_Size=50
LR=0.0001
num_classes=10
net=VGGnet(in_channels=1, num_classes=num_classes).to(device)
register_for_hook(net)
trainData=torchvision.datasets.MNIST(
    root="/home/lpx/codes/hook/data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)
 
train_loader=DataLoader(dataset=trainData,batch_size=Batch_Size,shuffle=True)
test_data=torchvision.datasets.MNIST(root="home/lpx/codes/hook/data",train=False,download=True)

def Train(model):
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
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            layer_list.inputs.clear()
            layer_list.outputs.clear()
            layer_list.layers.clear()
            layer_list.n = 0
            
    print('res finish training')

Train(net)