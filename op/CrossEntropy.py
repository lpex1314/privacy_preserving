import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess
forward, backward = getdll()
from globals import global_param
num_segments = global_param.num_segmentation

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

class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pre: Tensor, labels: Tensor, classes: int) -> Tensor:
        # print('**********cross_entropy***********')
        # for i in range(6):
        #     print(y_pre[i*num_segments])
        shape_x, x, len_x, double_array_x = preprocess(y_pre)
        shape_y, y, len_y, double_array_y = preprocess(labels)
        int_array_shape = c_int * 2
        N, C = shape_x[0] // num_segments, shape_x[1]
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
        print('*******CrossEntropyBackward*******')
        # print(gradOutput) [1.]
        y_pred, labels = ctx.saved_tensors  # y
        # print('loss:{}'.format(loss))
        classes = ctx.classes
        class_c = c_int(classes)
        shape_target, y_true, len_y, double_array_yt = preprocess(labels)  # target
        # shape_target [batch_size, num_classes]
        shape_y, y_pred_c, len_y, double_array_yp = preprocess(y_pred)  # y
        # shape_y [batch_size * num_segments, num_classes]
        N, C = shape_y[0] // num_segments, shape_target[1]
        shape_new = torch.Size([N, C])
        len_y = N * C
        result = torch.rand_like(y_pred)
        shape_res, result_c, len_res, double_array_res = preprocess(result)
        len_c = c_int(len_y)
        int_array_shape = c_int * 2
        shape_c = int_array_shape(*list(shape_new))
        # print('shape_ylabel', shape_target)
        # print('shape_ypred', y_pred.shape)
        # # print(labels)
        # print(y_pred.max())
        # print(y_pred.min())
        backward.d_crossentropy.argtypes = (double_array_yp, double_array_yt, c_int, c_int, double_array_res, int_array_shape)
        backward.d_crossentropy(y_pred_c, y_true, len_c, class_c, result_c, shape_c)
        output = np.frombuffer(result_c, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float)
        output = output.reshape(*shape_y)  # [N*num_segments, C]
        # print(output.max())
        # print(output.min())
        # print(output.shape)
        print(decrypt(output).max().item())
        return output, None, None


class CrossEntropy(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, y_pre: Tensor, labels: Tensor,classes: int) -> Tensor:
        return CrossEntropyFunction.apply(y_pre, labels, classes)
