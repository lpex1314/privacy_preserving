# linear_layers类用于存储模型运行过程中线性层的输入和模型
import torch
      

class linear_layers():
    def __init__(self) -> None:
        self.layers = []
        self.inputs = []
        self.n = 0
        pass
    
    def update(self, model, input):
        self.n += 1
        self.layers.append(model)
        self.inputs.append(input)

        
    def get_layer(self):
        layer = self.layers[self.n - 1]
        x = self.inputs[self.n - 1]
        self.n -= 1
        return layer, x
      
layer_list = linear_layers()
