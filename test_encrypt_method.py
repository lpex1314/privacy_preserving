# 这个python文件用来绘制概率分布函数，以验证加密方式2的理论正确性
from preprocess import getdll, preprocess
import torch
from ctypes import *
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
Ne = 10
forward, backward = getdll()
shape = torch.Size([10,3,28,28]) 
x = torch.randn(shape)
print(x.max())
print(x.min())
shape_res = torch.Size([10*Ne,3,28,28])
res = torch.randn(shape_res)

shape_x, x_c, len_x, double_array_x = preprocess(x)
shape_r, res_c, len_res, double_array_res = preprocess(res)

len_c = c_int(len_x)
forward.ecall_encrypt_test.argtypes = (double_array_x, c_int, double_array_res)
forward.ecall_encrypt_test(x_c, len_c, res_c)

cipher = np.frombuffer(res_c, dtype=np.double)
cipher = cipher.reshape(shape_res)
colors = ['k', 'w', 'b', 'r', 'g', 'k', 'w', 'b', 'r', 'g']
for j in range(Ne):
    data = cipher[j].flatten()
    
    ds_sort = sorted(data)
    last, i = min(ds_sort), 0
    while i < len(ds_sort):
        plt.plot([last, ds_sort[i]], [i/len(ds_sort), i/len(ds_sort)], c=colors[j], lw=2.5)
        if i < len(ds_sort):
            last = ds_sort[i]
        i += 1
    plt.grid()
plt.savefig("/home/lpx/codes/hook/pics/test.png")
 

