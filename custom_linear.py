import torch
from torch import nn, Tensor
from torch.autograd import Variable

def self_grad(input: Tensor, layer, cast):
    # x: Tensor = torch.ones_like(input, requires_grad=True)
    # x = x * input
    # x : Tensor = Variable(input.data, requires_grad=True)
    x = input.clone().detach()
    cast = cast.clone().detach()
    x.requires_grad = True
    print(x.requires_grad)
    # print(layer.weight)
    # if isinstance(layer, nn.Linear):
    func = nn.Linear(in_features=layer.in_features, out_features=layer.out_features, bias=True)
    func.load_state_dict(layer.state_dict())
    # elif isinstance(layer, nn.Conv2d):
    #     func = nn.Conv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, bias=True)
    #     func.load_state_dict(layer.state_dict())
    print(func)
    y = func(x)
    y = y.requires_grad_()
    print(y.grad_fn)
    print(layer)
    print(id(layer))
    print(id(func))
    # print(layer.weight == func.weight)
    dy = y * cast
    dy = dy.requires_grad_()
    print(dy.requires_grad)
    print(dy.sum())
    dy.sum().backward()
    dw = func.weight.grad.data
    return dw

input = torch.rand([50, 512], requires_grad=True)
m = nn.Linear(in_features=512, out_features=10, bias=True)
output = m(input)
print(output.is_leaf)
grad_out = torch.randn_like(output)
h = grad_out / output
output.sum().backward()
new_dw = self_grad(input, m, h)
print(new_dw)
# print(input.grad)
# print(conv1.weight.grad)
