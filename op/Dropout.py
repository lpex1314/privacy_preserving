import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess

forward, backward = getdll()


class _DropoutNd(nn.Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
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
    def forward(ctx, input: Tensor, p, training, inplace) -> Tensor:
        print('************dropout************')
        shape_x, x, len_x, double_array_x = preprocess(input)
        shape_x = list(shape_x)
        int_array = c_int * 4
        shape = int_array(*shape_x)
        forward.dropout.argtypes = (double_array_x, c_double, int_array)
        forward.dropout(x, p, shape)
        output = np.frombuffer(x, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput


class Dropout(_DropoutNd):

    def forward(self, input: Tensor) -> Tensor:
        return DropoutFunction.apply(input, self.p, self.training, self.inplace)
