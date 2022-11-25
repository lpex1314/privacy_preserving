from importlib.util import set_loader
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from replace import 
from op_nn.ReLU import ReLU
from op_nn.Dropout import Dropout
from op_nn.MaxPool2d import MaxPool2d
from op_nn.Softmax import Softmax
from op_nn.CrossEntropy import CrossEntropy
from op_nn.Encrypt import encrypt, decrypt
import numpy as np
from torch.autograd import Variable
from torchsummary import summary
from hooks.hook_test import register_hook
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.0001
momentum = 0.5
log_interval = 10
random_seed = 1
classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device={}'.format(device))
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# print(example_targets)
# print(example_data.shape)

# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
def linearF_(shape_0, x, f):
    res = []
    for i in range(shape_0):
        res.append(f(x[i]).detach().numpy().tolist())
    res = np.array(res)
    return torch.from_numpy(res)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.kernel_size = 2
        self.stride = 2
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=False)
        self.dropout = Dropout()
        self.fc1 = nn.Linear(320, 50, bias=False)
        self.fc2 = nn.Linear(50, 10, bias=False)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.maxpool2d(x))
        x = self.relu(self.maxpool2d(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.softmax(x)


network = Net()
network = network.to(device)
register_hook(network)
# network=hook(network)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# print(optimizer.param_groups)
# parm_group = summary(network, input_size=(1, 28, 28), batch_size=-1)
# print(parm_group)
# print(network.get_parameter('conv1'))
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
criterion = nn.CrossEntropyLoss()

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target) 
        # output.requires_grad_(True)
        # print(output.grad)
        # print(output.requires_grad)
        # loss = F.nll_loss(output, target)
        # loss_var = Variable(loss.data, requires_grad=True)
        # loss_var.backward()
        # loss.requires_grad = True
        # print(loss.grad)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                        len(train_loader.dataset),
                                                                        100. * batch_idx / len(train_loader),
                                                                        loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
        # torch.save(network.state_dict(), './model.pth')
        # torch.save(optimizer.state_dict(), './optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


train(1)

test()  # 不加这个，后面画图就会报错：x and y must be the same size
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
    output = network(example_data)
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# ----------------------------------------------------------- #

continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

for i in range(4, 9):
    test_counter.append(i * len(train_loader.dataset))
    train(i)
    test()

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()
