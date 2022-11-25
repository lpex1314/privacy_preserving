import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess

forward, backward = getdll()


class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        print('**********sigmoid***********')
        shape_x, x, len_x, double_array_x = preprocess(input)
        forward.sigmoid.argtypes = (double_array_x, c_int)
        forward.sigmoid(x, len_x)
        output = np.frombuffer(x, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        shape_x, x, len_x, double_array_x = preprocess(gradOutput)
        output = x
        backward.sigmoid.argtypes = (double_array_x, c_int, double_array_input)
        backward.sigmoid(x, len_x, output)
        output = np.frombuffer(output, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x)
        return output


class Sigmoid(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return SigmoidFunction.apply(input)
