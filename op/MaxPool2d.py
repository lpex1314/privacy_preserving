import torch
from typing import Optional
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess
from ctypes import *
from typing import TypeVar, Union, Tuple
forward, backward = getdll()
from globals import global_param
num_segments = global_param.num_segmentation
T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
size_2_t = _scalar_or_tuple_2_t[int]
size_any_t = _scalar_or_tuple_any_t[int]

class _MaxPool2d(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']
    return_indices: bool
    ceil_mode: bool

    def __init__(self, kernel_size: size_2_t, stride: Optional[size_2_t] = None,
                 padding: size_2_t = 0, dilation: size_2_t = 1,
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

def encrypt(x):
    shape_x = x.shape
    N = shape_x[0]
    res = torch.rand([N * num_segments, *shape_x[1:]])
    _, x, len_x, double_array_x = preprocess(x)
    print(len_x)
    shape_res, res_c, len_res, double_array_res = preprocess(res)
    forward.ecall_encrypt.argtypes = (double_array_x, c_int, double_array_res)
    forward.ecall_encrypt(x, c_int(len_x), res_c)
    input = np.frombuffer(res_c, dtype=np.double)
    input = torch.tensor(input, dtype=torch.float)
    input = input.reshape(*shape_res)  # [40, 7]
    return input

def decrypt(x):
    shape_out, out_c, len_out, double_array_out = preprocess(x)
    N = shape_out[0] // num_segments
    result = torch.rand(N, *shape_out[1:])
    shape_dec, res_c, len_res, double_array_res = preprocess(result)
    forward.ecall_decrypt.argtypes = (double_array_out, c_int, double_array_res)
    forward.ecall_decrypt(out_c, c_int(len_res), res_c)
    dec_out = np.frombuffer(res_c, dtype=np.double)
    dec_out = torch.tensor(dec_out, dtype=torch.float)
    return dec_out

class MaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride):
        # print('**********maxpool***********')
        shape_x, x, len_x, float_array_x = preprocess(input)
        # print(shape_x)
        int_array_shape = c_int * 4
        N, C, H, W = shape_x[0] // num_segments, shape_x[1], shape_x[3], shape_x[3]
        shape_x = torch.Size([N,C,H,W])
        shape_c = int_array_shape(*shape_x)
        H_out = 1 + (H - kernel_size) // stride
        W_out = 1 + (W - kernel_size) // stride
        len_out = N * C * H_out * W_out
        # print(len_out)
        len_x = N * C * H * W
        res = torch.rand([num_segments*len_out])
        shape_res, res_c, len_res, float_array_res = preprocess(res)
        len_x_c = c_int(len_x)
        len_out_c = c_int(len_out)
        shape_new = torch.Size([N * num_segments, C, H_out, W_out])
        arg_shape = torch.Size([N, C, H_out, W_out])
        int_array_maxarg = c_int * len_out
        maxarg = torch.randint(0, len_out, [len_out])
        maxarg = maxarg.cpu().numpy().tolist()
        maxarg = int_array_maxarg(*maxarg)
        forward.ecall_max_pool_2d.argtypes = (float_array_x, c_int, float_array_res, c_int, int_array_shape, c_int, c_int, int_array_maxarg)
        forward.ecall_max_pool_2d(x, len_x_c, res_c, len_out_c, shape_c, kernel_size, stride, maxarg)
        # print("maxpool OK!")
        result = np.frombuffer(res_c, dtype=np.double)
        result = torch.tensor(result, dtype=torch.float)
        result = result.reshape(*shape_new)
        # dec_result = np.frombuffer(x, dtype=np.double)
        # dec_result = torch.tensor(dec_result, dtype=torch.float)
        # dec_result = dec_result.reshape(*shape_x)
        # print('dec_result:{}'.format(dec_result))
        # print(dec_result.shape)
        # print('max_pool_result:{}'.format(result))
        arg_out = np.frombuffer(maxarg, dtype=np.int32)
        arg_out = torch.tensor(arg_out)
        ctx.save_for_backward(input, result, arg_out)
        ctx.H_out = H_out
        ctx.W_out = W_out
        ctx.stride = stride
        ctx.kernel_size = kernel_size
        # self.maxarg = arg_out.tolist()
        # print(result.max())
        # print(result.min())
        # print(result)
        return result

    @staticmethod
    def backward(ctx, gradOutput):
        print('*********maxpool_backward*********')
        print(gradOutput.shape)
        print(gradOutput)
        # print(gradOutput.max())
        # print(gradOutput.min())
        # print('maxpool back: grad_out max', dec_grad_out.max().item())
        input, output, maxarg = ctx.saved_tensors
        H_out, W_out, kernel_size, stride = ctx.H_out, ctx.W_out, ctx.kernel_size, ctx.stride
        shape_y, dy, len_y, float_array_dy = preprocess(gradOutput)
        shape_x = input.shape
        len_x = input.flatten().shape[0] // num_segments
        len_y = len_y // num_segments
        len_x_c = c_int(len_x)
        len_y_c = c_int(len_y)
        result = torch.randn(*shape_x)
        shape_res, res, len_res, float_array_res = preprocess(result)
        int_array_maxarg = c_int * len_x
        arg = int_array_maxarg(*maxarg.cpu().detach().numpy().tolist())
        backward.d_max_pool_2d.argtypes=(float_array_dy, float_array_res, c_int, c_int, int_array_maxarg)
        backward.d_max_pool_2d(dy, res, len_y_c, len_x_c, arg)
        # maxpool函数，传给TEE：只需dy，maxarg即可求导，不需要使用到y
        out = np.frombuffer(res, dtype=np.double)
        out = torch.tensor(out, dtype=torch.float)
        out = out.reshape(*shape_x)
        # print(out.max())
        # print(out.min())
        # print('gradOutput.shape:{}'.format(gradOutput.shape))
        print('self op', decrypt(out))
        return out, None, None

class MaxPool2d(_MaxPool2d):
    kernel_size: size_2_t
    stride: size_2_t
    padding: size_2_t
    dilation: size_2_t
    def forward(self, input: Tensor):
        return MaxPool2dFunction.apply(input, self.kernel_size, self.stride)
