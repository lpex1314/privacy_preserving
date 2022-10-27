from ctypes import *


def preprocess(x):
    shape = x.shape
    tmp = x.flatten()
    len = tmp.shape[0]
    double_array = c_double * len
    tmp = tmp.cpu().detach().numpy().tolist()
    tmp = double_array(*tmp)
    return shape, tmp, len, double_array


def getdll():
    return CDLL('dll/forward.so'), CDLL('dll/backward.so')
