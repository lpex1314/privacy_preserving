import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from torch.autograd import gradcheck

def preprocess(x):
    shape = x.shape
    tmp = x.flatten()
    len = tmp.shape[0]
    double_array = c_double * len
    tmp = tmp.cpu().detach().numpy().tolist()
    tmp = double_array(*tmp)
    return shape, tmp, len, double_array


def getdll():
    return cdll.LoadLibrary('dll/forward.so'), cdll.LoadLibrary('dll/backward.so')

def getdll2():
    return cdll.LoadLibrary('dll_enc2/forward.so'), cdll.LoadLibrary('dll_enc2/backward.so')

forward, backward = getdll2()
class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, training: bool, eps, momentum, running_mean, running_var, weight):
        shape_in, x_c, len_x, float_array_x = preprocess(x)
        res = torch.rand(*shape_in)
        shape_res, res_c, len_res, float_array_res = preprocess(res)
        # print(input.max())
        # print(input.min())
        N, C, H, W = shape_in[0] // 10, shape_in[1], shape_in[2], shape_in[3]
        shape_x = torch.Size([N,C,H,W])
        mid = torch.rand(*shape_x)
        shape_mid, mid_c, len_mid, double_array_mid = preprocess(mid)
        len_x = N*C*H*W
        shape_running_mean, mean_c, len_mean, float_array_mean = preprocess(running_mean)
        shape_running_var, var_c, len_var, float_array_var = preprocess(running_var)
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
            float_array_var, c_int, double_array_mid)
        forward.ecall_batchnorm2d(x_c, res_c, eps_c, momentum_c, mode_c, shape_c, mean_c, var_c, len_c, mid_c)
        # reassigned to moving mean, moving var
        input = np.frombuffer(res_c, dtype=np.double)
        input = torch.tensor(input, dtype=torch.float)
        input = input.reshape(*shape_in)
        mid = np.frombuffer(mid_c, dtype=np.double)
        mid = torch.tensor(mid, dtype=torch.float)
        mid = mid.reshape(*shape_x)
        # print('-'*10+'middle result of BN:'+'-'*10)
        # print(mid.max())
        # print(mid.min())
        moving_mean = np.frombuffer(mean_c, dtype=np.double)
        moving_mean = torch.tensor(moving_mean, dtype=torch.float)
        running_mean = moving_mean
        moving_var = np.frombuffer(var_c, dtype=np.double)
        moving_var = torch.tensor(moving_var, dtype=torch.float)
        running_var = moving_var
        output = input
        for i in range(input.shape[1]):
            output[:, [i]] = input[:, [i]] * weight[i] 
#       print(input.shape, running_mean.shape, running_var.shape)
        ctx.save_for_backward(input, weight, running_mean, running_var, torch.tensor(eps))
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        print(gradOutput.shape)
        x_hat, weight, running_mean, running_var, eps = ctx.saved_tensors
        shape_x = x_hat.shape
        N, C, H, W = shape_x[0] // 10, shape_x[1], shape_x[2], shape_x[3]
        shape_new = torch.Size([N,C,H,W])
        grad_input = torch.randn(shape_x, dtype=torch.double)
        grad_weight = torch.randn(C, dtype=torch.double)
        grad_bias = torch.randn(C, dtype=torch.double)
        shape_x, x_hat_c, len_x, double_array_x = preprocess(x_hat)
        shape_dy, dy_c, len_dy, double_array_dy = preprocess(gradOutput)
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
        grad_weight = np.frombuffer(dw_c, dtype=np.float64)
        grad_weight = torch.tensor(grad_weight, dtype=torch.double)
        # print('grad_w.shape:{}'.format(grad_weight.shape))
        # print('grad_x.shape:{}'.format(grad_input.shape))
        return grad_input, None, None, None, None, None, grad_weight


class BatchNorm(nn.Module):
    
    def __init__(self, channels):
        super(BatchNorm, self).__init__()
        self.channels = channels
        self.eps = 1e-05
        self.momentum = 0.1
        self.running_mean = torch.zeros([self.channels])
        self.running_var = torch.ones([self.channels])
        self.weight = torch.ones([self.channels], requires_grad=True)
        self.training = True
        self.kwargs = self.eps, self.momentum, self.running_mean, self.running_var
    def forward(self, input: Tensor) -> Tensor:
        return BatchNormFunction.apply(input, self.training, self.eps, self.momentum, self.running_mean, self.running_var, self.weight)



bn = BatchNormFunction.apply
running_mean = torch.zeros([3])
running_var = torch.ones([3])
weight = torch.ones([3], requires_grad=True)
test_input = (torch.rand([6,3,28,28], requires_grad=True), True, torch.tensor(1e-5), torch.tensor(0.1), running_mean, running_var, weight)
test = gradcheck(bn, test_input, eps=1e-6, atol=1e-4)
print(test)