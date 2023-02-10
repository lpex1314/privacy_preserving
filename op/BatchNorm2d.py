import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import init
# from deltas import delta_list
import warnings
from torch.nn.modules.utils import _single, _pair
import math
import copy
import numpy as np
from preprocess import getdll, preprocess
from ctypes import *
forward, backward = getdll()
import torch.autograd as autograd

def compare_bn(bn1, bn2):
    err = False
    if not torch.allclose(bn1.running_mean, bn2.running_mean):
        print('Diff in running_mean: {} vs {}'.format(
            bn1.running_mean, bn2.running_mean))
        err = True

    if not torch.allclose(bn1.running_var, bn2.running_var):
        print('Diff in running_var: {} vs {}'.format(
            bn1.running_var, bn2.running_var))
        err = True

    if bn1.affine and bn2.affine:
        if not torch.allclose(bn1.weight, bn2.weight):
            print('Diff in weight: {} vs {}'.format(
                bn1.weight, bn2.weight))
            err = True

        if not torch.allclose(bn1.bias, bn2.bias):
            print('Diff in bias: {} vs {}'.format(
                bn1.bias, bn2.bias))
            err = True

    if not err:
        print('All parameters are equal!')

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, training=True):
        super(BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        self.eps = eps
        self.training = training
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features), requires_grad=True)
        else:
            self.register_parameter('weight', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            # self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)


    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)



    def forward(self, x):
        return BatchNorm2dFunction.apply(x, self.weight, self.running_mean, self.running_var, self.eps, self.training, self.momentum)
  

class BatchNorm2dFunction(autograd.Function):

    """
    Autograd function for a linear layer with asymmetric feedback and feedforward pathways
    forward  : weight
    backward : weight_feedback
    bias is set to None for now
    """

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(ctx, x: Tensor, weight: Tensor, running_mean, running_var, eps, training, momentum):
        # print("*************BN forward*************")
        shape_in, x_c, len_x, float_array_x = preprocess(x)
        res = torch.rand(*shape_in)
        shape_res, res, len_res, float_array_res = preprocess(res)
        N, C, H, W = shape_in[0] // 10, shape_in[1], shape_in[2], shape_in[3]
        shape_x = torch.Size([N,C,H,W])
        mid = torch.rand(*shape_x)
        shape_mid, mid_c, len_mid, double_array_mid = preprocess(mid)
        len_x = N*C*H*W
        shape_running_mean, mean_c, len_mean, float_array_mean = preprocess(running_mean)
        shape_running_var, var_c, len_var, float_array_var = preprocess(running_var)
        shape_w, w_c, len_w, float_array_w = preprocess(weight)
        eps_c = c_double(eps)
        momentum_c = c_double(momentum)
        len_c = c_int(len_x)
        if training:
            mode = 1
        else:
            mode = 0
        mode_c = c_int(mode)
        int_array_shape = c_int * 4
        shape_c = int_array_shape(*shape_x)
        forward.ecall_batchnorm2d.argtypes=(float_array_x, float_array_res, c_double, c_double, c_int, int_array_shape, float_array_mean, \
            float_array_var, c_int, double_array_mid, float_array_w)
        forward.ecall_batchnorm2d(x_c, res, eps_c, momentum_c, mode_c, shape_c, mean_c, var_c, len_c, mid_c, w_c)
        output = np.frombuffer(res, dtype=np.double)
        output = torch.tensor(output, dtype=torch.float, requires_grad=True)
        output = output.reshape(*shape_in)
        # 可获取中间结果以debug
        mid = np.frombuffer(mid_c, dtype=np.double)
        mid = torch.tensor(mid, dtype=torch.float)
        mid = mid.reshape(*shape_x)
        # reassigned to moving mean, moving var
        # 这里计算的running_mean、running_var不能改变BatchNormmodule里的属性(此处无法修改self.running_mean)，需要再优化
        moving_mean = np.frombuffer(mean_c, dtype=np.double)
        moving_mean = torch.tensor(moving_mean, dtype=torch.float)
        running_mean = moving_mean
        moving_var = np.frombuffer(var_c, dtype=np.double)
        moving_var = torch.tensor(moving_var, dtype=torch.float)
        running_var = moving_var
        ctx.save_for_backward(x, weight, running_mean, running_var)
        ctx.eps = eps
        return output # y = batchmorm2d(x)

    @staticmethod
    def backward(ctx, grad_output):
        # print("************BNbackward************")
        x, weight, running_mean, running_var = ctx.saved_tensors
        eps = ctx.eps
        shape_x = x.shape
        N, C, H, W = shape_x[0] // 10, shape_x[1], shape_x[2], shape_x[3]
        shape_new = torch.Size([N,C,H,W])
        grad_input = torch.randn(shape_x, dtype=torch.double)
        grad_weight = torch.randn(C, dtype=torch.double)
        grad_bias = torch.randn(C, dtype=torch.double)
        shape_x, x_hat_c, len_x, double_array_x = preprocess(x)
        shape_dy, dy_c, len_dy, double_array_dy = preprocess(grad_output)
        shape_m, rmean_c, len_m, double_array_mean = preprocess(running_mean)
        shape_v, rvar_c, len_v, double_array_var = preprocess(running_var)
        shape_w, w_c, len_w, double_array_w = preprocess(weight)
        len_x = N * C * H * W
        eps_c = c_double(eps)
        len_c = c_int(len_x)
        shape_array = c_int * 4
        shape_c = shape_array(*shape_new)
        shape_dw, dw_c, len_dw, double_array_dw = preprocess(grad_weight)
        shape_db, db_c, len_db, double_array_db = preprocess(grad_bias)
        backward.d_batchnorm.argtypes = (double_array_x, double_array_dy, double_array_mean, double_array_var, double_array_w,\
            c_double, c_int, shape_array, double_array_dw, double_array_db)
        backward.d_batchnorm(x_hat_c, dy_c, rmean_c, rvar_c, w_c, eps_c, len_c, shape_c, dw_c, db_c)
        grad_input = np.frombuffer(x_hat_c, dtype=np.double)
        grad_input = torch.tensor(grad_input, dtype=torch.float)
        grad_input = grad_input.reshape(shape_x)
        # print('grad_x:{}'.format(grad_input[0].shape))
        grad_weight = np.frombuffer(dw_c, dtype=np.float64)
        grad_weight = torch.tensor(grad_weight, dtype=torch.double)
        grad_input.requires_grad = True
        grad_weight.requires_grad = True
        # print('grad_x:{}'.format(grad_input[0][0]))
        # print('grad_w.shape:{}'.format(grad_weight.shape))
        # print('grad_x.shape:{}'.format(grad_input.shape))
        return grad_input, grad_weight, None, None, None, None, None

