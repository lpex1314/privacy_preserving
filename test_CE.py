import torch
import numpy as np
from ctypes import *
from torch import nn
from preprocess import getdll2, preprocess
from op_enc2.CrossEntropy import CrossEntropy
Ne = 10
forward, backward = getdll2()
num_classes = 7
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
    input = torch.tensor(input, dtype=torch.float32)
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
    dec_out = torch.tensor(dec_out, dtype=torch.float32)
    return dec_out


x=torch.rand(4,num_classes)  # 4个样本，共7类
x1=x.clone()  # 4个样本，共7类
x2=x.clone()  # 4个样本，共7类
x1.requires_grad=True
x2.requires_grad=True
y=torch.LongTensor([1,3,5,0]) # 对应的标签
# torch 原生
criterion = torch.nn.CrossEntropyLoss()
out = criterion(x1,y)
out.backward()
print('pytorch库 loss:', out, 'grad:', x1.grad)

# 自定义op
x_ = encrypt(x2)
x_.requires_grad = True
target_one_hot = torch.zeros(4, num_classes).scatter(1, y.view(4,1), 1)
loss_func = CrossEntropy()
loss = loss_func(x_, target_one_hot, num_classes)
print('self op loss:', loss)
loss.backward()

