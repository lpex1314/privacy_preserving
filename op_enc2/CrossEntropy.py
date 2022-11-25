import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll2, preprocess
from deltas import delta_list
forward, backward = getdll2()


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pre: Tensor, labels: Tensor, classes: int) -> Tensor:
        # print('**********cross_entropy***********')
        # for i in range(6):
        #     print(y_pre[i*10])
        shape_x, x, len_x, double_array_x = preprocess(y_pre)
        shape_y, y, len_y, double_array_y = preprocess(labels)
        int_array_shape = c_int * 2
        N, C = shape_x[0] // 10, shape_x[1]
        shape_x = torch.Size([N, C])
        len_x = N * C
        # print('shape_x:{}'.format(shape_x))
        shape_c = int_array_shape(*shape_x)
        # tmp = torch.randn(input.shape)
        # shape_res, res, len_res, float_array_res = preprocess(tmp)
        res = torch.rand(2)
        float_array_res = c_double * 2
        result = float_array_res(*res)
        class_c = c_int(classes)
        len_c = c_int(len_x)
        forward.ecall_entropy.argtypes = (double_array_x, double_array_y, c_int, c_int, float_array_res, int_array_shape)
        forward.ecall_entropy(x, y, len_c, class_c, result, shape_c)
        loss = np.frombuffer(result, dtype=np.double)[0]
        loss = torch.tensor(loss, dtype=torch.float)
        ctx.save_for_backward(y_pre, labels)
        ctx.classes = classes
        # print('loss:{}'.format(loss))
        return loss

    @staticmethod
    def backward(ctx, gradOutput):
        # print('**********cross_entropy_backward***********')
        # print(gradOutput)
        y_pred, labels = ctx.saved_tensors  # y
        # print('loss:{}'.format(loss))
        classes = ctx.classes
        class_c = c_int(classes)
        shape_y, y_true, len_y, double_array_yt = preprocess(labels)  # target
        shape_y_pre, y_pred_c, len_y, double_array_yp = preprocess(y_pred)  # y
        N, C = shape_y[0] // 10, shape_y[1]
        shape_new = torch.Size([N, C])
        len_y = N * C
        result = torch.rand_like(y_pred)
        shape_res, result_c, len_res, double_array_res = preprocess(result)
        len_c = c_int(len_y)
        int_array_shape = c_int * 2
        shape_c = int_array_shape(*list(shape_new))
        # print(y_pred)
        # print(labels)
        # print(y_pred.max())
        # print(y_pred.min())
        backward.d_crossentropy.argtypes = (double_array_yt, double_array_yp, c_int, c_int, double_array_res, int_array_shape)
        backward.d_crossentropy(y_true, y_pred_c, len_c, class_c, result_c, shape_c)
        output = np.frombuffer(result_c, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_y_pre)  # [N*10, C]
        # print(output.max())
        # print(output.min())
        # print(output.shape)
        return output, None, None


class CrossEntropy(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, y_pre: Tensor, labels: Tensor,classes: int) -> Tensor:
        return CrossEntropyFunction.apply(y_pre, labels, classes)
