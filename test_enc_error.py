# 该函数将原文和加密再解密后的数值相减, 观察加解密前后的数值误差
import torch
from preprocess import getdll2, preprocess
from ctypes import *
import numpy as np
forward, backward = getdll2()
shape = torch.Size([50, 1, 28, 28])
Ne = 10
x = torch.rand(shape)
N, C, H, W = x.shape
res = torch.rand([N * Ne, C, H, W])
shape_x, x_c, len_x, double_array_x = preprocess(x)
shape_res, res_c, len_res, double_array_res = preprocess(res)
forward.ecall_encrypt.argtypes = (double_array_x, c_int, double_array_res)
forward.ecall_encrypt(x_c, c_int(len_x), res_c)
cipher = np.frombuffer(res_c, dtype=np.double)
cipher = torch.tensor(cipher, dtype=torch.float)
cipher = cipher.reshape(*shape_res)  # [500, 1, 28, 28]

plain = torch.rand_like(x)
shape_cipher, cipher_c, _, double_array_cipher = preprocess(cipher)
shape_plain, plain_c, _, double_array_plain = preprocess(plain)
forward.ecall_decrypt.argtypes = (double_array_cipher, c_int, double_array_plain)
forward.ecall_decrypt(cipher_c, c_int(len_x), plain_c)
plaintext = np.frombuffer(plain_c, dtype=np.double)
plaintext = torch.tensor(plaintext, dtype=torch.float)
plaintext = plaintext.reshape(*shape_plain)
print(x - plaintext)