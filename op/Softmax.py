import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess
from typing import List, Optional
from globals import global_param
num_segments = global_param.num_segmentation
forward, backward = getdll()


class SoftmaxFunction(torch.autograd.Function):
    dim: Optional[int]
    
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:   
         # print('**********softmax***********')
        shape_x_hat, x_hat, len_hat, double_array_xhat = preprocess(input)
        N, C = shape_x_hat[0] // num_segments, shape_x_hat[1]
        len_x = N * C
        shape_res = torch.Size([N, C])
        res = torch.rand(shape_res)
        shape_res, result, len_res, double_array_res = preprocess(res)
        int_array_shape = c_int * 2
        len_c = c_int(len_x)
        shape_c = int_array_shape(*shape_res)
        forward.ecall_softmax_easy.argtypes = (double_array_xhat, c_int, double_array_res, int_array_shape)
        forward.ecall_softmax_easy(x_hat, len_c, result, shape_c)
        output = np.frombuffer(result, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_x_hat)
        # print(output)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        print('**********softmax_backward***********')
        # print(gradOutput.max())
        # print(gradOutput.min())
        output, = ctx.saved_tensors
        res = torch.rand_like(output)
        shape_y_hat, y, len_y, double_array_y_hat = preprocess(output)
        shape_dy_hat, dy, len_dy, double_array_dy_hat = preprocess(gradOutput)
        shape_res, result, len_res, double_array_res = preprocess(res)
        N, C = shape_y_hat[0] // num_segments, shape_y_hat[1]
        int_array_shape = c_int * 2
        shape_x = torch.Size([N, C])
        len_x = N * C
        len_c = c_int(len_x)
        shape_c = int_array_shape(*shape_x)
        backward.d_softmax_easy.argtypes = (double_array_dy_hat, double_array_y_hat, double_array_res, c_int, int_array_shape)
        backward.d_softmax_easy(dy, y, result, len_c, shape_c)
        grad_in = np.frombuffer(result, dtype=np.double)
        grad_in = torch.tensor(grad_in, dtype=torch.float)
        grad_in = grad_in.reshape(*shape_y_hat)
        return grad_in


class Softmax(nn.Module):

    def __init__(self) -> None:
        super(Softmax, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        # print(delta1.shape)
        return SoftmaxFunction.apply(input)

