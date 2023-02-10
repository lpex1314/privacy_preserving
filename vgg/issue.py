# All necessary imports at the beginning
import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

# make a vgg model
VGG_types = {
"VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,512,512,512],
}
net = torchvision.models.vgg16(pretrained=True)
# this function aims
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


    
def register_for_hook(model):
    for _, i in model.named_modules():
        if isinstance(i, nn.Linear):  
            i.register_forward_hook(forward_hook)
            i.register_backward_hook(backward_hook) 


# here is a custom weight-computation function, the 'input' is the input of a specific layer in a pytorch model
# and the 'layer' is exactly the layer, the 'cast' is the ratio of grad_output to output, the grad_output is the
# partial derivative of loss wrt to output, the output is the output of the specific layer, in my case, the layer
# is always nn.Conv2d or nn.Linear
def compute_grad(input: Tensor, layer: nn.Linear, cast : Tensor):
    x = input.clone().detach()
    cast = cast.clone().detach()
    x.requires_grad = True
    
    # Create an identical linear layer without backward
    func = nn.Linear(in_features=layer.in_features, out_features=layer.out_features, bias=True)
    # Import parameter dictionary
    func.load_state_dict(layer.state_dict())
    
    y = func(x)
    # The anomaly is: the grad_fn of y is None, but the attribute of is_leaf is True
    print(y.grad_fn)
    # out:None
    print(y.is_leaf)
    # out:True
    dy = y * cast
    dy.sum().backward()
    # 这块报错RuntimeError: element 0 of tensors does not require grad
    # and does not have a grad_fn
    dw = func.weight.grad
    print(dw)
    return dw

def forward_hook(module, input, output):
    # the forward_hook is used to record the inputs, outputs and layers of all linear layers
    # in a pytorch model during running.
    # In order to reduce the amount of code, there is no need to list in detail
    return 
    
def backward_hook(module, grad_in, grad_out):
    # the backward_hook is used to 
    # 1. Obtain the input, output and layer that we recorded in the forward_hook
    # 2. Since the input and output were encrypted, so we decrypt them
    # 3. Pass the input, layer and gout/output to our compute_grad func to compute real grad
    # new_dw = compute_grad(input=input, layer=module, cast=grad_out/output)
    # return new_dw to replace the original dw
    return 

device = 'cpu' if torch.cuda.is_available() else 'cpu'
Epoch=3
Batch_Size=50
LR=0.0001
num_classes=10
net=VGGnet(in_channels=1, num_classes=num_classes).to(device)
# register hook for the net
register_for_hook(net)

# datasets
trainData=torchvision.datasets.MNIST(
    root="/home/lpx/codes/hook/data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)
 
train_loader=DataLoader(dataset=trainData,batch_size=Batch_Size,shuffle=True)
test_data=torchvision.datasets.MNIST(root="home/lpx/codes/hook/data",train=False,download=True)

def Train(model):
    # Run the regular training process, the loss will go throught the backward process
    # and trigger the backward hook.
    # loss.backward()
    # In order to reduce the amount of code, there is no need to list in detail
    return
Train(net)


'''
Traceback (most recent call last):
  File "vgg.py", line 159, in <module>
    Train(net)
  File "vgg.py", line 158, in Train
    loss.backward()
  File "/home/lpx/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/tensor.py", line 195, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/home/lpx/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/autograd/__init__.py", line 97, in backward
    Variable._execution_engine.run_backward(
  File "vgg.py", line 132, in backward_hook
    new_dw = compute_grad(input, layer, cast)
  File "vgg.py", line 114, in self_grad
    dy.sum().backward()
  File "/home/lpx/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/tensor.py", line 195, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/home/lpx/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/autograd/__init__.py", line 97, in backward
    Variable._execution_engine.run_backward(
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
'''