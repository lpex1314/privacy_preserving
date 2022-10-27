import torch
from model.AlexNet import AlexNet
from hook import hook

device = torch.device('cpu')
model = AlexNet()
model.to(device)
model = hook(model)
img = torch.rand((1, 3, 224, 224))
pred = model(img)
