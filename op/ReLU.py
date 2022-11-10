import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess

forward, backward = getdll()


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, delta1: Tensor, delta2: Tensor) -> Tensor:
        # print('**********relu***********')
        shape_x, x, len_x, double_array_x = preprocess(input)
        shape_fr, f_r, len_r, float_array_f_r = preprocess(delta1)
        shape_delta, delta_c, len_del, float_array_delta = preprocess(delta2)
        tmp = torch.randn(input.shape)
        shape_res, res, len_res, float_array_res = preprocess(tmp)
        forward.ecall_relu.argtypes = (double_array_x, float_array_f_r, float_array_res, c_int, float_array_delta)
        forward.ecall_relu(x, f_r, res, len_x, delta_c)
        output = np.frombuffer(res, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        ctx.save_for_backward(output, delta1, delta2)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        print('**********relu_backward***********')
        # print(gradOutput)
        print(gradOutput.max() )
        print(gradOutput.min() )
        y, delta1, delta2 = ctx.saved_tensors  # y
        shape_y, y, len_y, double_array_y = preprocess(y)  # y
        shape_dy, dy, len_dy, double_array_dy = preprocess(gradOutput)  # dy
        shape_del, delta2_c, len_del, double_array_del = preprocess(delta2)
        output = dy
        len_c = c_int(len_y)
        backward.d_relu.argtypes = (double_array_dy, double_array_y, c_int, double_array_dy, double_array_del)
        backward.d_relu(dy, y, len_c, output, delta2_c)
        output = np.frombuffer(output, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_dy)
        return output, None, None


class ReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor, delta1: Tensor, delta2: Tensor) -> Tensor:
        return ReLUFunction.apply(input, delta1, delta2)
