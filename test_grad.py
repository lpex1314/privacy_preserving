from op_nn.BatchNorm2d import BatchNorm
import torch
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()
        
def compare_bn(bn1, bn2):
    err = False
    # print(bn1.running_mean)
    # print(bn2.running_mean)
    if not torch.allclose(bn1.running_mean, bn2.running_mean):
        print('Diff in running_mean: {} vs {}'.format(
            bn1.running_mean, bn2.running_mean))
        err = True

    if not torch.allclose(bn1.running_var, bn2.running_var):
        print('Diff in running_var: {} vs {}'.format(
            bn1.running_var, bn2.running_var))
        err = True

    if bn1.affine and bn2.affine:
        if not torch.allclose(bn1.weight, bn2.weight):
            print('Diff in weight: {} vs {}'.format(
                bn1.weight, bn2.weight))
            err = True

        if not torch.allclose(bn1.bias, bn2.bias):
            print('Diff in bias: {} vs {}'.format(
                bn1.bias, bn2.bias))
            err = True

    if not err:
        print('All parameters are equal!')

my_bn = BatchNorm(3, affine=True) 
bn = torch.nn.BatchNorm2d(3, affine=True)

hookB_bn = Hook(bn,backward=True) 
hookB_mybn = Hook(my_bn,backward=True)

compare_bn(my_bn, bn)  # weight and bias should be different
# Load weight and bias
my_bn.load_state_dict(bn.state_dict())
compare_bn(my_bn, bn)

x = torch.randn(2, 3, 2, 2) 
o2 = bn(x)
o1 = my_bn(x)

o2.backward(torch.ones_like(o1), retain_graph=True)
o1.backward(torch.ones_like(o2), retain_graph=True)

print('***'*3+'  nn.bn Backward Hooks Inputs & Outputs shapes  '+'***'*3)
print('input:', [i.shape for i in hookB_bn.input if hasattr(i, 'shape')])
print('len input:{}'.format(len(hookB_bn.input)))
print(hookB_bn.input[0])
print('output:' ,[i.shape for i in hookB_bn.output if hasattr(i, 'shape')])  
print('---'*17)

print('***'*3+'  mybn Backward Hooks Inputs & Outputs shapes  '+'***'*3)
print('input:',[i.shape for i in hookB_mybn.input if hasattr(i, 'shape')])
print('len input:{}'.format(len(hookB_mybn.input)))
print('output:' ,[i.shape for i in hookB_mybn.output if hasattr(i, 'shape')])    
# print(hookB_mybn.input[1], hookB_mybn.input[2])
# print(hookB_mybn.input[0])
print('---'*17)

print('***'*3+'  outputs are the same  '+'***'*3)
# print(hookB_mybn.output[0]==hookB_bn.output[0])

print('***'*3+'  inputs differ  '+'***'*3)
print(hookB_mybn.input[0]==hookB_bn.input[0])

# my_bn = BatchNorm(3, affine=True) 
# my_bn = my_bn.double()
# x = torch.randn(1, 2, 3, 3, dtype=torch.double,requires_grad=True)
# print(torch.autograd.gradcheck(my_bn, x))