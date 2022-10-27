import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess
from typing import List, Optional

forward, backward = getdll()


class SoftmaxFunction(torch.autograd.Function):
    dim: Optional[int]

    @staticmethod
    def forward(self, ctx, input: Tensor, dim) -> Tensor:
        print('**********softmax***********')
        self.dim = dim
        shape_x, x, len_x, double_array_x = preprocess(input)
        int_array_shape = c_int * 4
        dim = c_int(self.dim)
        output = x
        forward.softmax.argtypes = (double_array_x, int_array_shape, c_int, double_array_x)
        forward.softmax(x, shape_x, dim, output)
        output = np.frombuffer(output, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        return output

    @staticmethod
    def backward(self, ctx, gradOutput):
        dim = c_int(self.dim)
        shape_x, x, len_x, double_array_x = preprocess(gradOutput)
        output = x
        N = shape_x[0]
        C = shape_x[1]
        H = shape_x[2]
        W = shape_x[3]
        double_array = c_double * (C * H * W * (N * N))
        backward.softmax.argtypes = (double_array_x, double_array_x, c_int, c_int, double_array_x)
        backward.softmax(x, output, len_x, dim, shape_x)
        output = np.frombuffer(output, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        return output


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

    def forward(self, input: Tensor) -> Tensor:
        return SoftmaxFunction(input, self.dim)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)
