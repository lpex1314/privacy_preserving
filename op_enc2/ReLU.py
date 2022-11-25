import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll2, preprocess
from deltas import delta_list
from hooks.Encipher import encipher

forward, backward = getdll2()


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        # print('**********relu***********')
        # print(input.max())
        # print(input.min())
        shape_x, x, len_x, double_array_x = preprocess(input)
        res = torch.rand(*shape_x)
        N, C, H, W = shape_x[0] // 10, shape_x[1], shape_x[2], shape_x[3]
        shape_new = torch.Size([N,C,H,W])
        shape_res, res_c, len_res, double_array_res = preprocess(res)
        len_x = len_x // 10
        mid = torch.rand([len_x])
        shape_mid, mid_c, len_mid, double_array_mid = preprocess(mid)
        len_c = c_int(len_x)
        forward.ecall_relu.argtypes = (double_array_x, c_int, double_array_res, double_array_mid)
        forward.ecall_relu(x, len_c, res_c, mid_c)
        mid = np.frombuffer(mid_c, dtype=np.double)
        mid = torch.tensor(mid, dtype=torch.float)
        mid = mid.reshape(*shape_new)
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
        # print('**********relu_backward***********')
        # print(gradOutput)
        # print(gradOutput.max())
        # print(gradOutput.min())
        y, = ctx.saved_tensors  # y
        shape_y, y, len_y, double_array_y = preprocess(y)  # y
        shape_dy, dy, len_dy, double_array_dy = preprocess(gradOutput)  # dy
        N, C, H, W = shape_y[0] // 10, shape_y[1], shape_y[2], shape_y[3]
        len_new = N * C * H * W
        len_c = c_int(len_new)
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
