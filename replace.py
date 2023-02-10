from torch import nn
from op.ReLU import ReLU
from op.MaxPool2d import MaxPool2d
from op.BatchNorm2d import BatchNorm
from op.Dropout import Dropout
from op.Softmax import Softmax
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
            elif isinstance(module, nn.Softmax):
                softmax = Softmax()
                _set_module(model, name, softmax)
            elif isinstance(module, nn.BatchNorm2d):
                channel = module.num_features
                bn_ = BatchNorm(channel, training=True)
                _set_module(model, name, bn_)
            elif isinstance(module, nn.Linear):
                linear = nn.Linear(module.in_features, module.out_features, bias=False)
                _set_module(model, name, linear)
            elif isinstance(module, nn.Conv2d):
                conv = nn.Conv2d(module.in_channels, module.out_channels, kernel_size=module.kernel_size, stride=\
                    module.stride, padding=module.padding, bias=False)
                _set_module(model, name, conv)
    return model
