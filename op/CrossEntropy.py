import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess

forward, backward = getdll()


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pre: Tensor, labels: Tensor, delta1: Tensor, classes: int) -> Tensor:
        # print('**********relu***********')
        shape_x, x, len_x, double_array_x = preprocess(y_pre)
        shape_y, y, len_y, double_array_y = preprocess(labels)
        shape_fr, delta1_c, len_r, float_array_del1 = preprocess(delta1)
        # tmp = torch.randn(input.shape)
        # shape_res, res, len_res, float_array_res = preprocess(tmp)
        res = torch.rand(2)
        float_array_res = c_double * 2
        result = float_array_res(*res)
        class_c = c_int(classes)
        forward.ecall_entropy.argtypes = (double_array_x, double_array_y, c_int, c_int, float_array_del1, float_array_res)
        forward.ecall_entropy(x, y, len_x, class_c, delta1_c, result)
        loss = np.frombuffer(result, dtype=np.double)[0]
        loss = torch.tensor(loss, dtype=torch.float)
        ctx.save_for_backward(loss, y_pre, labels, delta1)
        ctx.classes = classes
        return loss

    @staticmethod
    def backward(ctx, gradOutput):\
        # gradOutput=1? Yes! 1 = dl/dl
        print('**********cross_entropy_backward***********')
        # print(gradOutput)
        loss, y_pred, labels, delta1 = ctx.saved_tensors  # y
        classes = ctx.classes
        class_c = c_int(classes)
        shape_fr, delta1_c, len_r, float_array_del1 = preprocess(delta1)
        shape_y, y_true, len_y, double_array_yt = preprocess(labels)  # y
        shape_y_pre, y_pre, len_y, double_array_yp = preprocess(y_pred)  # dy
        output = y_pre
        len_c = c_int(len_y)
        int_array_shape = c_int * 2
        shape_c = int_array_shape(*list(shape_y_pre))
        # print(y_pred)
        # print(labels)
        print(y_pred.max())
        print(y_pred.min())
        backward.d_crossEntropy.argtypes = (double_array_yt, double_array_yp, int_array_shape, c_int, double_array_yp, c_int, float_array_del1)
        backward.d_crossEntropy(y_true, y_pre, shape_c, len_c, output, class_c, delta1_c)
        output = np.frombuffer(output, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_y_pre)
        print(output.max())
        print(output.min())
        # print(output.shape)
        return output, None, None, None, None


class CrossEntropy(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, y_pre: Tensor, labels: Tensor, delta1: Tensor, classes: int) -> Tensor:
        return CrossEntropyFunction.apply(y_pre, labels, delta1, classes)
