import torch
import numpy as np
from ctypes import *
from torch import nn
from preprocess import getdll, preprocess
from op.MaxPool2d import MaxPool2d
Ne = 10
forward, backward = getdll()
def encrypt(x):
    shape_x = x.shape
    N = shape_x[0]
    res = torch.rand([N * Ne, *shape_x[1:]])
    _, x, len_x, double_array_x = preprocess(x)
    print(len_x)
    shape_res, res_c, len_res, double_array_res = preprocess(res)
    forward.ecall_encrypt.argtypes = (double_array_x, c_int, double_array_res)
    forward.ecall_encrypt(x, c_int(len_x), res_c)
    input = np.frombuffer(res_c, dtype=np.double)
    input = torch.tensor(input, dtype=torch.float)
    input = input.reshape(*shape_res)  # [40, 7]
    return input

def decrypt(x):
    shape_out, out_c, len_out, double_array_out = preprocess(x)
    N = shape_out[0] // Ne
    result = torch.rand(N, *shape_out[1:])
    shape_dec, res_c, len_res, double_array_res = preprocess(result)
    forward.ecall_decrypt.argtypes = (double_array_out, c_int, double_array_res)
    forward.ecall_decrypt(out_c, c_int(len_res), res_c)
    dec_out = np.frombuffer(res_c, dtype=np.double)
    dec_out = torch.tensor(dec_out, dtype=torch.float)
    dec_out = dec_out.reshape(shape_dec)
    return dec_out


shape = torch.Size([2, 1, 6, 6])
x = torch.rand(shape)
x1 = x.clone()
x2 = x.clone()
x1.requires_grad = True
x2.requires_grad = True
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
y1 = maxpool(x1)
y1.sum().backward()
# print('torch op, x1.grad:', x1.grad.shape, '\n', x1.grad)

x_ = encrypt(x2)
x_.requires_grad = True
m2 = MaxPool2d(kernel_size=2, stride=2)
y2 = m2(x_)
y2.sum().backward()
y2_ = decrypt(y2)
print(y2_)
print('y1-y2',y1 - y2_)


