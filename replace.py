from torch import nn
from op_enc2.ReLU import ReLU
from op_enc2.Sigmoid import Sigmoid
from op_enc2.Dropout import Dropout
from op_enc2.MaxPool2d import MaxPool2d
from op_enc2.Softmax import Softmax
from op_enc2.BatchNorm2d import BatchNorm


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    setattr(cur_mod, tokens[-1], module)


def register_op(model):
    for name, module in model.named_modules():
        # print(name)
        if type(module) != nn.Sequential:
            if isinstance(module, nn.Dropout):
                dropout = Dropout(p=module.p, inplace=module.inplace)
                _set_module(model, name, dropout)
            elif isinstance(module, nn.MaxPool2d):
                maxpool = MaxPool2d(kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                                    dilation=module.dilation, return_indices=module.return_indices,
                                    ceil_mode=module.ceil_mode)
                _set_module(model, name, maxpool)
            elif isinstance(module, nn.ReLU):
                relu = ReLU(inplace=module.inplace)
                _set_module(model, name, relu)
            elif isinstance(module, nn.Sigmoid):
                sigmoid = Sigmoid()
                _set_module(model, name, sigmoid)
            elif isinstance(module, nn.Softmax):
                softmax = Softmax(dim=module.dim)
                _set_module(model, name, softmax)
            elif isinstance(module, nn.BatchNorm2d):
                channel = module.num_features
                bn = BatchNorm(channel, affine=True)
                _set_module(model, name, bn)
    return model
