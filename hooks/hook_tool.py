from torch import nn
from hooks.Encipher import encipher
from preprocess import getdll, preprocess
import torch
from ctypes import *
import numpy as np
forward, backward = getdll()
def isLinear(feature):
    if isinstance(feature, nn.Linear) or isinstance(feature, nn.Conv2d):
        return True
    else:
        return False

def register_hook(module):
    for _, i in module.named_modules():
        if isLinear(i):
            # print(i)
            i.register_backward_hook(backward_hook2)

def backward_hook2(module, gin, gout):
    print(module)
    # print("len:"+str(len(gout)))
    # print(gout[0].shape)
    if len(gin) > 2:
        gin0, gin1, gin2 = gin
        # print("gin0.shape:{}".format(gin0.shape)) 
        # print("gin1.shape:{}".format(gin1.shape)) 
        if isinstance(module, nn.Linear):
            gin0, gin2, gin1 = dec_linear(gin2, gin0, gin1, gout)
        elif isinstance(module, nn.Conv2d):
            gin2, gin1, gin0 = dec_conv(gin1, gin2, gin0, gout)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            print("get wrong")
        new_gin = tuple([gin0, gin1, gin2]) # db, dw, dx
    else:
        gin0, gin1 = gin #  = dx, dw
        print("dwbefore:" + str(gin1.flatten()[0:10]))
        # print("gin0.shape:{}".format(gin0.shape)) 
        # print("gin1.shape:{}".format(gin1.shape)) 
        if isinstance(module, nn.Linear):
            _, gin1, gin0 = dec_linear(gin1, None, gin0, gout)
        else:
            gin2, gin1, gin0 = dec_conv(gin1, None, gin0, gout)
        print("dw:" + str(gin1.flatten()[0:10]))
        print("w:" + str(module.weight.flatten()[0:10]))
        # print("grad:"+str(torch.mean(gout[0])))
        # print("dw:"+str(torch.mean(gin1)))
        new_gin = tuple([gin0, gin1])  # dx, dw
    return new_gin


def dec_linear(dw, db, dx, gout):
    # print("***********linear_backward**********")
    # print(dx.max())
    # print(dx.min())
    grad_out = gout[0]
    a = 1
    # print("a:" + str(a))
    new_db, new_dx = None, None
    if db is not None:
        new_db = db / a
    if dx is not None:
        new_dx = dx 
    delta = encipher.get_delta()
    # print('n:{}'.format(encipher.n))
    shape_del, delta_c, len_delta, float_array_delta = preprocess(delta)
    shape_gout, gout_c, len_gout, float_array_gout = preprocess(grad_out)
    int_array_shape = c_int * 4
    shape_c = int_array_shape(*shape_gout)
    shape_dw, dw_c, _, float_array_w = preprocess(dw)
    shape_x, _, len_x, _ = preprocess(dx)
    # print('len_x:{}'.format(len_x))
    # print('dw.shape:{}'.format(dw.shape))
    # print('dx.shape:{}'.format(dx.shape))
    # print('dy.shape:{}'.format(grad_out.shape))
    len_c = c_int(len_x)
    backward.decrypt_linear.argtypes = (float_array_delta, float_array_gout, float_array_w, c_int, int_array_shape)
    backward.decrypt_linear(delta_c, gout_c, dw_c, len_c, shape_c)
    new_dw = np.frombuffer(dw_c, dtype=np.double)
    new_dw = torch.tensor(new_dw, dtype=torch.float)
    new_dw = new_dw.reshape(*shape_dw)
    # print("tmp:"+str(tmp.flatten()[0:10]))
    # print("pre_dw:" + str(dw.flatten()[0:10]))
    return new_db, new_dw, new_dx

def dec_conv(dw, db, dx, gout):
    # print("***********conv_backward**********")
    # print(dx.max())
    # print(dx.min())
    grad_out = gout[0]
    a = 1
    # print("a:" + str(a))
    new_db, new_dx = None, None
    if db is not None:
        new_db = db / a
    if dx is not None:
        new_dx = dx 
    delta = encipher.get_delta()
    # print('n:{}'.format(encipher.n))
    
    shape_del, delta_c, len_delta, float_array_delta = preprocess(delta)
    shape_gout, gout_c, len_gout, float_array_gout = preprocess(grad_out)
    shape_dw, dw_c, _, float_array_w = preprocess(dw)
    # print('dw.shape:{}'.format(dw.shape))
    # print('dx.shape:{}'.format(dx.shape))
    # print('dy.shape:{}'.format(grad_out.shape))
    shape_x, _, len_x, _ = preprocess(dx)
    
    # print('len_x:{}'.format(len_x))
    batch_size = shape_gout[0]
    h_in = shape_x[2]
    h_out = shape_gout[2]
    int_array_shape = c_int * 4
    shape_c = int_array_shape(*shape_dw)
    
    len_c = c_int(len_x)
    h_in = c_int(h_in)
    h_out = c_int(h_out)
    batch_size = c_int(batch_size)
    backward.decrypt_conv.argtypes = (float_array_w, float_array_delta, float_array_gout, c_int, int_array_shape, c_int, c_int, c_int)
    backward.decrypt_conv(dw_c, delta_c, gout_c, len_c, shape_c, h_in, h_out, batch_size)
    new_dw = np.frombuffer(dw_c, dtype=np.double)
    new_dw = torch.tensor(new_dw, dtype=torch.float)
    new_dw = new_dw.reshape(*shape_dw)
    # print("tmp:"+str(tmp.flatten()[0:10]))
    # print("pre_dw:" + str(dw.flatten()[0:10]))
    # new_dw = (dw - torch.sum(tmp[key], dim=0).to(device).T)
    return new_db, new_dw, new_dx

