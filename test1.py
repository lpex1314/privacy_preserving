from importlib.util import set_loader
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hook import hook
from op.ReLU import ReLU
from op.Dropout import Dropout
from op.MaxPool2d import MaxPool2d
from op.Softmax import Softmax
from op.CrossEntropy import CrossEntropy
from op.Encrypt import encrypt, decrypt
import numpy as np
from torch.autograd import Variable
from torchsummary import summary
from hooks.hook_tool import register_hook
from hooks.Encipher import encipher
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
        self.N_ks = 16
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=False)
        self.dropout = Dropout()
        self.fc1 = nn.Linear(320, 10, bias=False)
        # self.fc2 = nn.Linear(50, 10)
        self.relu = ReLU()
        self.maxpool2d = MaxPool2d(self.kernel_size, self.stride)
        self.softmax = Softmax(dim=1)
        self.encrypt = encrypt
        self.decrypt = decrypt
    def forward(self, x, delta_init):
        # encrypt
        # print(x.requires_grad)
        # self.delta_0 = torch.rand(self.N_ks, *x.shape)
        # x = self.encrypt(x, self.delta_0)  # x+r
        # print(x.requires_grad)
        self.delta_0 = delta_init
        # x = self.fctest(x)
        # x_enc = self.fctest(x_enc)
        # r = linearF_(self.delta_0.shape[0], self.delta_0, self.fctest) 
        
        # conv1
        print(x.requires_grad)
        x = self.conv1(x)  # f(x)
        # print(x.requires_grad)
        r = linearF_(self.delta_0.shape[0], self.delta_0, self.conv1)   # f(r)
        # maxpool2d
        # print('x:{}'.format(x))  # f(x)
        N, C, H, W = x.shape[0], x.shape[1],x.shape[2], x.shape[3]
        H_out = 1 + (H - self.kernel_size) // self.stride
        W_out = 1 + (W - self.kernel_size) // self.stride
        shape_pool = torch.Size([N, C, H_out, W_out])
        self.delta_1 = torch.rand(self.N_ks, *shape_pool, requires_grad=False) * 0.01
        # encipher.deltas['delta_maxpool_1'] = self.delta_1
        x = self.maxpool2d(x, r, self.delta_1)
        # print(x.requires_grad)
        # relu
        self.delta_2 = torch.rand(self.N_ks, *x.shape, requires_grad=False) * 0.01
        x = self.relu(x, self.delta_1, self.delta_2)
        #  conv2
        encipher.deltas.append(self.delta_2)
        x, r = self.conv2(x), linearF_(self.delta_2.shape[0], self.delta_2, self.conv2)
        #  dropout
        self.delta_3 = torch.rand(self.N_ks, *x.shape, requires_grad=False) * 0.01
        # encipher.deltas['dropout'] = self.delta_3
        x = self.dropout(x, r, self.delta_3)
        #  maxpool2d
        N, C, H, W = x.shape[0], x.shape[1],x.shape[2], x.shape[3]
        H_out = 1 + (H - self.kernel_size) // self.stride
        W_out = 1 + (W - self.kernel_size) // self.stride
        shape_pool = torch.Size([N, C, H_out, W_out])
        self.delta_o = torch.rand(self.N_ks, *shape_pool, requires_grad=False) * 0.01
        # encipher.deltas['maxpool_2'] = self.delta_o
        x = self.maxpool2d(x, self.delta_3, self.delta_o)
        #  relu
        self.delta_4 = torch.rand(self.N_ks, *x.shape, requires_grad=False) * 0.01
        x = self.relu(x, self.delta_o, self.delta_4)
        x = x.view(-1, 320)
        self.delta_4 = self.delta_4.view(self.N_ks, -1, 320)
        
        # print(x.requires_grad)
        #  fc
        encipher.deltas.append(self.delta_4)
        x, r = self.fc1(x), linearF_(self.delta_4.shape[0], self.delta_4, self.fc1)
        #  relu
        self.delta_5 = torch.rand(self.N_ks, *x.shape, requires_grad=False) * 0.01
        x = self.relu(x, r, self.delta_5)
        # print(x.requires_grad)
        #  softmax
        self.delta_6 = torch.rand(self.N_ks, *x.shape, requires_grad=False) * 0.01
        x = self.softmax(x, self.delta_5, self.delta_6)
        # decrypt
        # print(x.requires_grad)
        # x = self.decrypt(x, self.delta_6)
        return x, self.delta_6


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
criterion = CrossEntropy()

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.to(device)
        optimizer.zero_grad()
        delta_init = torch.rand(16, *data.shape, requires_grad=False) * 0.01
        encipher.deltas.append(delta_init)
        # print(delta_init.max())
        # print(data.max())
        # print(data.min())
        data_enc = encrypt(data, delta=delta_init)
        # data.requires_grad = True
        data_enc.requires_grad = True
        output, delta_dec = network(data_enc, delta_init)
        target_one_hot = torch.zeros(64, 10).scatter(1, target.view(64,1),1) 
        # print(target_one_hot)
        # the criterion will decrypt first and then compute loss
        loss = criterion(output, target_one_hot, delta_dec, classes) 
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
