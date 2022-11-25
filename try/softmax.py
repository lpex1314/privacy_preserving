import torch
import numpy as np
a = torch.tensor([1.,2.,3.,4.,5.], requires_grad=True) 
m = torch.nn.Softmax(dim=0)
b = m(a)
print(b)  # b = softmax(a)
# b = b.numpy()
# grad = np.diag(b) - np.dot(b.T, b)
# print(grad)
b.backward(torch.ones_like(b))
print(a.grad)



