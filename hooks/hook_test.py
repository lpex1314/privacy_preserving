from torch import nn
from hooks.Encipher import encipher
from preprocess import getdll, preprocess
import torch
from ctypes import *
import numpy as np

def isLinear(feature):
    if isinstance(feature, nn.Linear) or isinstance(feature, nn.Conv2d):
        return True
    else:
        return False

def register_hook(module):
    for _, i in module.named_modules():
        if not isinstance(i, nn.Sequential):
            if isLinear(i):
                # print(i)
                i.register_backward_hook(backward_hook2)
            else:
                i.register_backward_hook(backward_hook3)

def backward_hook2(module, gin, gout):
    print(module)
    # print("len:"+str(len(gout)))
    # print(gout[0].shape)
    if len(gin) > 2:
        gin0, gin1, gin2 = gin
        print("dw:" + str(gin1.flatten()[0:10]))
        print("w:" + str(module.weight.flatten()[0:10]))
        new_gin = tuple([gin0, gin1, gin2])
    else:
        gin0, gin1 = gin
        print("dw:" + str(gin1.flatten()[0:10]))
        print("w:" + str(module.weight.flatten()[0:10]))
        # print("grad:"+str(torch.mean(gout[0])))
        # print("dw:"+str(torch.mean(gin1)))
        new_gin = tuple([gin0, gin1])  # dx, dw
    return new_gin

def backward_hook3(module, gin, gout):
    print(module)
    gin = gin[0]
    gout = gout[0]
    print(gin.shape)
    print(gout.shape)
    print("grad_input:" + str(gin.flatten()[0:10]))
    print("grad_output:" + str(gout.flatten()[0:10]))