import torch
class Delta():
    def __init__(self) -> None:
        self.deltas = []
        self.cur_layer = 0
        self.length = 16 * 64 * 3 * 28 * 28
        for i in range(50):
            self.deltas.append(torch.rand([self.length]))
        pass
    
    def get_delta(self, add=False):
        tmp = self.deltas[self.cur_layer]
        if add:
            self.cur_layer += 1
        return tmp
    
    def set_delta(self, new):
        self.deltas[self.cur_layer] = new
        
delta_list = Delta()
# a = delta_list.get_delta(add=False)
# new_shape = torch.Size([64, 1, 24, 24])
# a = a.resize_(16, *new_shape)
# delta_list.set_delta(a)
# b = delta_list.get_delta(add=False)
# print(b.shape)
# print(delta_list.cur_layer)