import torch
from ctypes import *
from torch import Tensor
import numpy as np
import torch.nn as nn
from preprocess import getdll, preprocess

forward, backward = getdll()


def encrypt(x, delta):
    shape_x, x_c, len_x, double_array_x = preprocess(x)
    shape_delta, delta_c, len_delta, double_array_delta = preprocess(delta)
    len_c = c_int(len_x)
    forward.ecall_encrypt.argtypes = (double_array_delta, double_array_x, c_int)
    forward.ecall_encrypt(delta_c, x_c, len_c)
    output = np.frombuffer(x_c, dtype=np.double)
    output = torch.tensor(output, dtype=torch.float)
    output = output.reshape(*shape_x)
    return output


def decrypt(x, delta):
    shape_x, x_c, len_x, double_array_x = preprocess(x)
    shape_delta, delta_c, len_delta, double_array_delta = preprocess(delta)
    len_c = c_int(len_x)
    forward.ecall_decrypt.argtypes = (double_array_x, double_array_delta, c_int)
    forward.ecall_decrypt(x_c, delta_c, len_c)
    output = np.frombuffer(x_c, dtype=np.double)
    output = torch.tensor(output, dtype=torch.float)
    output = output.reshape(*shape_x)
    return output