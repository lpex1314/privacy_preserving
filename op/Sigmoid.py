# Simoid没有用于经典模型所以自定义op未开发，但已开发cpp函数
import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess

forward, backward = getdll()


class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor):
        print('**********sigmoid***********')
        return 

    @staticmethod
    def backward(ctx, gradOutput):
        return 


class Sigmoid(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return SigmoidFunction.apply(input)
