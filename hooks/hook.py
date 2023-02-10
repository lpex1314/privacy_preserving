import torch.nn as nn
import torch
import numpy as np
from storage import layer_list
from ctypes import *
from torch import Tensor
from globals import global_param
from preprocess import preprocess, getdll
forward, backward = getdll()
num_segments = global_param.num_segmentation
index = [6, 2, 2, 3, 0, 6, 6, 7, 1, 2, 2, 4, 5, 0, 8, 4]
insert = 9
index.append(insert)


def isLinear(feature):
    if isinstance(feature, nn.Linear) or isinstance(feature, nn.Conv2d):
        return True
    else:
        return False
    

def register_for_hook(model):
    for _, i in model.named_modules():
        if not isinstance(i, nn.Sequential):
            if isLinear(i):  # 如果是线性层
                # print(i)
                i.register_forward_hook(trace_hook)
                i.register_backward_hook(backward_hook) 
            i.register_forward_hook(print_hook)
            


def decrypt(x):
    shape_out, out_c, len_out, double_array_out = preprocess(x)
    N = shape_out[0] // num_segments
    result = torch.rand(N, *shape_out[1:])
    shape_dec, res_c, len_res, double_array_res = preprocess(result)
    forward.ecall_decrypt.argtypes = (double_array_out, c_int, double_array_res)
    forward.ecall_decrypt(out_c, c_int(len_res), res_c)
    dec_out = np.frombuffer(res_c, dtype=np.double)
    dec_out = torch.tensor(dec_out, dtype=torch.float)
    return dec_out
    
def trace_hook(module, input, output):
    layer_list.update(module, input)
    
def print_hook(module, input, output):
    dec_out = decrypt(output)
    # print(module, 'output max ', dec_out.max().item())


def compute_grad(input: Tensor, layer, grad_out : Tensor):
    x = input.clone().detach()
    x.requires_grad = True
    # print(x.shape)
    # # [50, 512] 50 is batch_size
    grad_out = grad_out.clone().detach()
    if isinstance(layer, nn.Linear):
        func = nn.Linear(in_features=layer.in_features, out_features=layer.out_features, bias=True)
    else:
        layer: nn.Conv2d
        func = nn.Conv2d(in_channels=layer.in_channels, out_channels=layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride\
            ,padding=layer.padding)
    func.weight = layer.weight
    # w1 = func.weight.clone().detach()
    batch_size = x.shape[0] // 10
    grad_w = torch.zeros_like(func.weight)
    # print('grad_out max', grad_out.max().item())
    for i in range(len(index)):
        # x_list.append(x[i:i+batch_size])
        # y_list.append(func(x_list[i]))
        # grad_out_list.append(grad_out[i:i+batch_size])
        # grad_w_list.append(torch.autograd.grad(y_list[i], func.weight, grad_out_list[i])[0])
        # grad_w += grad_w_list[i]
        for j in range(len(index)):
            # print(func.weight.grad)
            # there are 10 xi, get x[i]
            x_slice = x[index[i] * batch_size : (index[i] + 1) * batch_size]
            # print(x_slice.shape)
            y_slice = func(x_slice)
            # print('y_slice max', y_slice.max().item())
            # print('y_slice grad_fn:{}'.format(y_slice.grad_fn))
            grad_out_slice = grad_out[index[j] * batch_size : (index[j] + 1) * batch_size]
            # print('grad_out_slice max', grad_out_slice.max().item())
            # print('weight max', func.weight.max().item())
            grad_slice = torch.autograd.grad(y_slice, func.weight, grad_out_slice, create_graph=True, retain_graph=True)[0]
            # print(func.weight.grad)
            # print('grad_slice max', grad_slice.max().item())
            grad_w += grad_slice
    # w2 = func.weight.clone().detach()
    # 验证梯度计算不会影响weight.data
    # print('w1-w2:{}'.format(w1-w2))
    # y = func(x)
    # grad_w = torch.autograd.grad(y, func.weight, grad_out)
    # print(grad_w.flatten()[:10])
    return grad_w


def backward_hook(module, gin, gout):
    layer, input = layer_list.get_layer()
    input = input[0]
    dy = gout[0]
    dy_dec = decrypt(dy)
    # print(module, 'grad out max ', dy_dec.max().item())
    # print(dy.shape)
    # print(module)
    # print(len(gin))
    if len(gin) > 2:
        if isinstance(layer, nn.Linear):
            db, dx, dw = gin
        else:
            dx, dw, db = gin
    else:
        dx, dw = gin
    # dx, dw = gin
    dx: Tensor
    # this dw is computed by encrypted dy and x, so it's not right
    dw: Tensor
    
    # print(len(gin))
    # 3
    # print(input.shape, module.bias.shape, module.weight.shape)
    # torch.Size([50, 512]) torch.Size([10]) torch.Size([10, 512])
    # print(dx.shape, db.shape, dw.shape)
    # 第一层dx为None
    # torch.Size([50, 512]) torch.Size([10]) torch.Size([512, 10])

    # call my custom compute_grad function
    with torch.enable_grad():
        self_dw = compute_grad(input, module, dy)
    # print(self_dw.shape)
    # print(dw.shape)
    # print(((self_dw.T - dw)))
    # assert self_dw.T == dw
    # if isinstance(module, nn.Linear):
    #     new_gin = tuple([ dx, self_dw.T])
    # elif isinstance(module, nn.Conv2d):
    #     new_gin = tuple([dx, self_dw, db])
    if isinstance(module, nn.Linear):
        self_dw = self_dw.T
    # In Linear layer, the dw.shape is the transpose of module.weight.shape, the pytorch will transpose the
    # shape of weight during actual computation
    if len(gin) > 2:
        if isinstance(layer, nn.Linear):
            new_gin = tuple([db, dx, self_dw])
        else:
            new_gin = tuple([dx, self_dw, db])
    else:
        new_gin = tuple([dx, self_dw])
    return new_gin
