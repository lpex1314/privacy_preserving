import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess
from typing import List, Optional
m = nn.MaxPool2d
forward, backward = getdll()


class SoftmaxFunction(torch.autograd.Function):
    dim: Optional[int]
    
    @staticmethod
    def forward(ctx, input: Tensor, dim, delta1, delta2) -> Tensor:   
    #   def forward(ctx, input: Tensor, dim) -> Tensor:
        # print('**********softmax***********')
        shape_x, x, len_x, double_array_x = preprocess(input)
        shape_fr, f_r, len_r, float_array_f_r = preprocess(delta1)
        shape_delta, delta_c, len_del, float_array_delta = preprocess(delta2)
        tmp = torch.randn(input.shape)
        shape_res, res, len_res, float_array_res = preprocess(tmp)
        int_array_shape = c_int * 2
        dim_c = c_int(dim)
        len_c = c_int(len_x)
        output = x
        shape_c = int_array_shape(*shape_x)
        forward.ecall_softmax.argtypes = (double_array_x, float_array_f_r, float_array_res, c_int, int_array_shape, c_int, float_array_delta)
        forward.ecall_softmax(x, f_r, res, len_c, shape_c, dim_c, delta_c)
        output = np.frombuffer(res, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        # print(output)
        ctx.dim = dim
        ctx.save_for_backward(output, delta1, delta2)
        # print(output.requires_grad)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        print('**********softmax_backward***********')
        print(gradOutput.max())
        print(gradOutput.min())
        dim = ctx.dim
        out, delta1, delta2 = ctx.saved_tensors
        shape_del2, delta2_c, len_del, double_array_del2 = preprocess(delta2)
        shape_x, y, len_x, double_array_y = preprocess(out)
        shape_dy, dy, len_dy, double_array_dy = preprocess(gradOutput)
        # print(shape_dy)
        # print(shape_x)
        output = y
        # print(shape_x)
        N = shape_x[0]
        C = shape_x[1]
        double_array_res = c_double * len_dy
        int_array_shape = c_int * 2
        result = torch.randn([len_dy])
        len_x_c = c_int(len_x)
        dim_c = c_int(dim)
        shape_c = int_array_shape(*shape_x)
        result_c = double_array_res(*result.cpu().detach().numpy().tolist())
        backward.d_softmax_easy.argtypes = (double_array_dy, double_array_y, double_array_res, c_int, c_int, int_array_shape, double_array_del2)
        backward.d_softmax_easy(dy, y, result_c, len_x_c, dim_c, shape_c, delta2_c)
        output = np.frombuffer(result_c, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_dy)
        print(output.max())
        print(output.min())
        return output, None, None, None


class Softmax(nn.Module):
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor, delta1: Tensor, delta2: Tensor) -> Tensor:
        # print(delta1.shape)
        return SoftmaxFunction.apply(input, self.dim, delta1, delta2)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)
