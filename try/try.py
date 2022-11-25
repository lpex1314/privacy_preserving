import torch
from torch import nn

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.autograd as autograd

import warnings
from torch.nn.modules.utils import _single, _pair
import math
import copy

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

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

#     def _check_input_dim(self, input):
#         raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)



    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
            
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        
        kwargs = self.training, bn_training, exponential_average_factor,self.track_running_stats 
        return BatchNorm2dFunction.apply(input,self.weight, self.bias, self.running_mean, self.running_var ,  self.eps, kwargs)

def Hadamard(one, two):
    """
    @author: hughperkins
    """
    # if one.size() != two.size():
    #     raise Exception('size mismatch %s vs %s' % (str(list(one.size())), str(list(two.size()))))
    # print('one:',one.shape, 'two', two.shape)
    try:
        one.view_as(two)
    except:
        if len(two.shape) ==1:
    
            two = two[None, :, None, None]
        two.expand_as(one)
    
    res = one * two
    assert res.numel() == one.numel()
    return res
    
      
    

class BatchNorm2dFunction(autograd.Function):

    """
    Autograd function for a linear layer with asymmetric feedback and feedforward pathways
    forward  : weight
    backward : weight_feedback
    bias is set to None for now
    """

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, bias, running_mean, running_var, eps, kwargs):
        
        training, bn_training, exponential_average_factor,track_running_stats = kwargs
        
#         print(input.shape, running_mean.shape, running_var.shape)
        input_hat = (input - running_mean[None, :, None, None])/torch.sqrt(running_var[None, :, None, None] + eps)
        input_hat.requires_grad = False
        context.save_for_backward(input,weight, bias, input_hat, running_mean, running_var, Variable(torch.tensor(eps)))
        
        
        
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean if not training or track_running_stats else None,
            running_var if not training or track_running_stats else None,
            weight, bias, bn_training, exponential_average_factor, eps)

    @staticmethod
    def backward(context, grad_output):
        input,  weight, bias, input_hat, running_mean, running_var, eps = context.saved_tensors
        eps = eps.item()
        N = input.shape[0]
        
        
        grad_weight = torch.einsum('bijk,bijk->ijk', input_hat, grad_output)
        grad_bias = torch.einsum('bijk,bijk->ijk', torch.ones_like(input_hat), grad_output)

        coef_inp = Hadamard((1/N)*weight, (running_var + eps)**(-0.5))
        part1 = -Hadamard(input_hat, grad_weight)
        part2 = N*grad_output
        part3 = -torch.einsum('nijk,oijk->nijk', torch.ones_like(input), grad_bias[None,:]).squeeze()

        if len(coef_inp.shape) ==1:
            coef_inp = coef_inp.unsqueeze(1).unsqueeze(2)
        else:
            coef_inp = coef_inp[None,:]

        grad_input = coef_inp.expand_as(part1) * (part1 + part2 + part3)

        return grad_input, grad_weight.sum(dim=(-1, -2)), grad_bias.sum(dim=(-1, -2)), None, None, None, None

my_bn = BatchNorm2d(3, affine=True).double() # MyBatchNorm2d(3, affine=True)

bn = nn.BatchNorm2d(3, affine=True).double()
x = torch.randn(1, 3, 7, 7,dtype=torch.double, requires_grad=True)
s=torch.autograd.gradcheck(my_bn, x)
print(s)