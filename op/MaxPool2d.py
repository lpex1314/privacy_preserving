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


class _MaxPool2d(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']
    return_indices: bool
    ceil_mode: bool

    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
                 padding: _size_any_t = 0, dilation: _size_any_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(_MaxPool2d, self).__init__()
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
    def forward(ctx, input, delta1, delta2, kernel_size, stride):
        # print('**********maxpool***********')
        shape_x, x, len_x, float_array_x = preprocess(input)
        shape_fr, delta1_c, len_r, float_array_delta1 = preprocess(delta1)
        shape_delta2, delta2_c, len_del, float_array_delta2 = preprocess(delta2)
        int_array_shape = c_int * 4
        shape_c = int_array_shape(*shape_x)
        N, C, H, W = shape_x[0], shape_x[1], shape_x[3], shape_x[3]
        H_out = 1 + (H - kernel_size) // stride
        W_out = 1 + (W - kernel_size) // stride
        len_out = N * C * H_out * W_out
        len_x_c = c_int(len_x)
        len_out_c = c_int(len_out)
        shape_new = [N, C, H_out, W_out]
        input_ = torch.randn([shape_x[0], shape_x[1], H_out,  W_out])
        shape_in, input_, len_in, float_array_input = preprocess(input_)
        int_array_maxarg = c_int * len_out
        maxarg = torch.randint(0, len_out, [len_out])
        maxarg = maxarg.cpu().numpy().tolist()
        maxarg = int_array_maxarg(*maxarg)
        forward.ecall_max_pool_2d.argtypes = (float_array_x, c_int, c_int, float_array_delta1, int_array_shape, c_int, c_int, float_array_input, float_array_delta2, int_array_maxarg)
        forward.ecall_max_pool_2d(x, len_x_c, len_out_c, delta1_c, shape_c, kernel_size, stride, input_, delta2_c, maxarg)
        result = np.frombuffer(input_, dtype=np.double)
        result = torch.tensor(result, dtype=torch.float)
        result = result.reshape(*shape_new)
        
        dec_result = np.frombuffer(x, dtype=np.double)
        dec_result = torch.tensor(dec_result, dtype=torch.float)
        dec_result = dec_result.reshape(*shape_x)
        # print('dec_result:{}'.format(dec_result))
        # print(dec_result.shape)
        # print('max_pool_result:{}'.format(result))
        arg_out = np.frombuffer(maxarg, dtype=np.int32)
        arg_out = torch.tensor(arg_out)
        agr_out = arg_out.reshape(*shape_new)
        ctx.save_for_backward(input, result, arg_out, delta1, delta2)
        ctx.H_out = H_out
        ctx.W_out = W_out
        ctx.stride = stride
        ctx.kernel_size = kernel_size
        # self.maxarg = arg_out.tolist()
        return result

    @staticmethod
    def backward(ctx, gradOutput):
        print('**********maxpool_backward***********')
        print(gradOutput.max())
        print(gradOutput.min())
        input, output, maxarg, delta1, delta2 = ctx.saved_tensors
        H_out, W_out, kernel_size, stride = ctx.H_out, ctx.W_out, ctx.kernel_size, ctx.stride
        shape_y, dy, len_y, float_array_y = preprocess(gradOutput)
        shape_x = input.shape
        len_x = input.flatten().shape[0]
        len_x_c = c_int(len_x)
        len_y_c = c_int(len_y)
        result = torch.randn(input.shape)
        shape_res, res, len_res, float_array_res = preprocess(result)
        int_array_maxarg = c_int * len_x
        arg = int_array_maxarg(*maxarg.cpu().detach().numpy().tolist())
        backward.d_max_pool_2d.argtypes=(float_array_y, float_array_res, c_int, c_int, int_array_maxarg)
        backward.d_max_pool_2d(dy, res, len_y_c, len_x_c, arg)
        # maxpool函数，传给TEE：dy，maxarg即可求导，不需要使用到y
        out = np.frombuffer(res, dtype=np.double)
        out = torch.tensor(out, dtype=torch.float)
        out = out.reshape(*shape_x)
        print(out.max())
        print(out.min())
        # print('gradOutput.shape:{}'.format(gradOutput.shape))
        return out, None, None, None, None, None

    
class MaxPool2d(_MaxPool2d):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t
    def forward(self, input: Tensor, delta1: Tensor, delta2: Tensor):
        return MaxPool2dFunction.apply(input, delta1, delta2, self.kernel_size, self.stride)
