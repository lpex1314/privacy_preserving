import torch.nn as nn
import torch

def register_hook(module):
    for _, i in module.named_modules():
        if not isinstance(i, nn.Sequential):
            if isinstance(i, nn.ReLU):
                i.register_forward_hook(relu_hook)
            if isinstance(i, nn.BatchNorm2d):
                i.register_forward_hook(BN_hook)
                
                
def relu_hook(model, input, output):
    print('middle result of nn relu')
    print(output.max())
    print(output.min())
    
    
def BN_hook(model, input, output):
    print('middle result of nn BN')
    print(output.max())
    print(output.min())