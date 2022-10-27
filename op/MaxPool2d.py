import torch
from typing import Optional
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess
from ctypes import *

forward, backward = getdll()
from torch.nn.common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
                                   _ratio_3_t, _ratio_2_t, _size_any_opt_t, _size_2_opt_t, _size_3_opt_t)


class _MaxPoolNd(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']
    return_indices: bool
    ceil_mode: bool

    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
                 padding: _size_any_t = 0, dilation: _size_any_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


class MaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride):
        print('**********maxpool***********')
        shape_x, x, len_x, float_array_x = preprocess(input)
        int_array_shape = c_int * 4
        shape_c = int_array_shape(*shape_x)
        N, C, H, W = shape_x[0], shape_x[1], shape_x[3], shape_x[3]
        H_out = 1 + (H - kernel_size) // stride
        W_out = 1 + (W - kernel_size) // stride
        shape_new = [N, C, H_out, W_out]
        shape_new = torch.tensor(shape_new)
        input = torch.randn([shape_x[0], shape_x[1], H_out, W_out])
        shape_in, input, len_in, float_array_input = preprocess(input)
        forward.max_pool_2d.argtypes = (float_array_x, int_array_shape, c_int, c_int, float_array_input)
        forward.max_pool_2d(x, shape_c, kernel_size, stride, input)
        output = np.frombuffer(input, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_new)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput


class MaxPool2d(_MaxPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: Tensor):
        return MaxPool2dFunction.apply(input, self.kernel_size, self.stride)
