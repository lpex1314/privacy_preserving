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
    def forward(ctx, input: Tensor, p: float, delta1, delta2) -> Tensor:
        print('************dropout************')
        shape_x, x, len_x, double_array_x = preprocess(input)
        shape_fr, delta1_c, len_r, float_array_delta1 = preprocess(delta1)
        shape_delta2, delta2_c, len_del, float_array_delta2 = preprocess(delta2)
        tmp = torch.randn(input.shape)
        shape_res, res, len_res, float_array_res = preprocess(tmp)
        len_c = c_int(len_x)
        shape_x = list(shape_x)
        int_array = c_int * 4
        shape = int_array(*shape_x)
        p_c = c_double(p)
        forward.ecall_dropout.argtypes = (double_array_x, float_array_delta1, c_int, float_array_res ,c_double, int_array, float_array_delta2)
        forward.ecall_dropout(x, delta1_c, len_c, res, p_c, shape, delta2_c)
        result = np.frombuffer(res, dtype=np.double)
        result = torch.tensor(result, dtype=torch.float)
        result = result.reshape(*shape_x)
        # print('result of dropout:{}'.format(result))
        ctx.p = p
        ctx.save_for_backward(result, delta1, delta2)
        return result

    @staticmethod
    def backward(ctx, gradOutput):
        print('************dropout_backward************')
        output, delta1, delta2 = ctx.saved_tensors
        p = ctx.p
        shape_dy, dy, len_dy, double_array_dy = preprocess(gradOutput)
        shape_out, out, len_out, double_array_out = preprocess(output)
        shape_del2, delta2_c, len_del, double_array_del = preprocess(delta2)
        result = torch.randn(shape_out)
        shape_output, result, len_output, double_array_output = preprocess(result)
        shape_out = list(shape_out)
        int_array = c_int * 4
        shape = int_array(*shape_out)
        # print(shape_out)
        len = shape_out[0] * shape_out[1] * shape_out[2] * shape_out[3]
        len_c = c_int(len)
        p = c_double(p)
        backward.d_dropout.argtypes = (double_array_dy, double_array_out, c_int, c_double, int_array, double_array_output, double_array_del)
        backward.d_dropout(dy, out, len_c, p, shape, result, delta2_c)
        gradout = np.frombuffer(result, dtype=np.double)
        gradout = torch.tensor(gradout, dtype=torch.float)
        gradout = gradout.reshape(*shape_out)
        # need gradOutput * gradout ?
        return gradout, None, None, None


class Dropout(_DropoutNd):

    def forward(self, input: Tensor,delta1: Tensor, delta2: Tensor) -> Tensor:
        return DropoutFunction.apply(input, self.p, delta1, delta2)
