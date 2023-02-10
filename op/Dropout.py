import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess
from globals import global_param
num_segments = global_param.num_segmentation
forward, backward = getdll()


class _DropoutNd(nn.Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.3, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, p: float) -> Tensor:
        # print('************dropout************')
        shape_x, x, len_x, double_array_x = preprocess(input)
        res = torch.rand_like(input)
        shape_res, res_c, len_res, double_array_res = preprocess(res)
        N,C,H,W = shape_x[0] // num_segments, shape_x[1], shape_x[2], shape_x[3]
        shape_real = torch.Size([N,C,H,W])
        len_ = N*C*H*W
        len_c = c_int(len_)
        int_array = c_int * 4
        shape_c = int_array(*shape_real)
        p_c = c_double(p)
        forward.ecall_dropout.argtypes = (double_array_x, c_int, c_double, int_array, double_array_res)
        forward.ecall_dropout(x, len_c, p_c, shape_c, res_c)
        result = np.frombuffer(res_c, dtype=np.double)
        result = torch.tensor(result, dtype=torch.float)
        result = result.reshape(*shape_x)
        # print('result of dropout:{}'.format(result))
        ctx.p = p
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, gradOutput):
        print('************dropout_backward************')
        print(gradOutput.max().item())
        output, = ctx.saved_tensors
        result = torch.rand_like(output)
        p = ctx.p
        shape_dy, dy, len_dy, double_array_dy = preprocess(gradOutput)
        shape_y, out, len_out, double_array_out = preprocess(output)
        shape_output, result_c, len_output, double_array_output = preprocess(result)
        N,C,H,W = shape_y[0] // num_segments, shape_y[1], shape_y[2], shape_y[3]
        shape_real = torch.Size([N,C,H,W])
        int_array = c_int * 4
        shape_c = int_array(*shape_real)
        # print(shape_out)
        len_ = N*C*H*W
        len_c = c_int(len_)
        p = c_double(p)
        backward.d_dropout.argtypes = (double_array_dy, double_array_out, c_int, c_double, int_array, double_array_output)
        backward.d_dropout(dy, out, len_c, p, shape_c, result_c)
        gradout = np.frombuffer(result_c, dtype=np.double)
        gradout = torch.tensor(gradout, dtype=torch.float)
        gradout = gradout.reshape(*shape_dy)
        return gradout, None


class Dropout(_DropoutNd):

    def forward(self, input: Tensor) -> Tensor:
        return DropoutFunction.apply(input, self.p)
