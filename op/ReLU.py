import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess
from globals import global_param
num_segments = global_param.num_segmentation
forward, backward = getdll()

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


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        # print('**********relu***********')
        # print(input.max())
        # print(input.min())
        shape_x, x, len_x, double_array_x = preprocess(input)
        res = torch.rand(*shape_x)
        shape_res, res_c, len_res, double_array_res = preprocess(res)
        len_x = len_x // num_segments
        mid = torch.rand([len_x])
        shape_mid, mid_c, len_mid, double_array_mid = preprocess(mid)
        len_c = c_int(len_x)
        forward.ecall_relu.argtypes = (double_array_x, c_int, double_array_res, double_array_mid)
        forward.ecall_relu(x, len_c, res_c, mid_c)
        output = np.frombuffer(res_c, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        ctx.save_for_backward(output)
        # print('-'*10 + "middle result relu" + '-'*10)
        # print(mid.max())
        # print(mid.min())
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        dec_grad_out = decrypt(gradOutput)
        print('relu back: grad_out max', dec_grad_out.max().item())
        # print('**********relu_backward***********')
        # print(gradOutput)
        # print(gradOutput.max())
        # print(gradOutput.min())
        y, = ctx.saved_tensors  # y
        shape_y, y, len_y, double_array_y = preprocess(y)  # y
        shape_dy, dy, len_dy, double_array_dy = preprocess(gradOutput)  # dy
        len_new = len_y // num_segments
        len_c = c_int(len_new)
        # print(shape_y)
        # print(len_c)
        result = torch.rand(*shape_dy)
        shape_res, res_c, len_res, double_array_res = preprocess(result)
        backward.d_relu.argtypes = (double_array_dy, double_array_y, c_int, double_array_res)
        backward.d_relu(dy, y, len_c, res_c)
        output = np.frombuffer(res_c, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_dy)
        # print(output.max())
        # print(output.min())
        return output


class ReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return ReLUFunction.apply(input)
