import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess

forward, backward = getdll()


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        print('**********relu***********')
        shape_x, x, len_x, double_array_x = preprocess(input)
        forward.relu.argtypes = (double_array_x, c_int)
        forward.relu(x, len_x)
        output = np.frombuffer(x, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        shape_x, x, len_x, double_array_x = preprocess(gradOutput)
        output = x
        backward.relu.argtypes = (double_array_x, c_int, double_array_x)
        backward.relu(x, len_x, output)
        output = np.frombuffer(output, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        return output


class ReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return ReLUFunction.apply(input)
