"""
Demo script for loading the dataset in batches
"""

from inaturalist_dataset import INaturalistDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

data_dir = '../data2/'
annotations_dir = '../annotations/'
train_annotations = '{}train2017_min.json'.format(annotations_dir)
val_annotations = '{}val2017.json'.format(annotations_dir)

inaturalist = INaturalistDataset(data_dir, train_annotations, transform=transforms.ToTensor())
all_ids = inaturalist.all_ids
# images, targets = inaturalist.get_images(all_ids)

batch_size = 10
train_loader = torch.utils.data.DataLoader(inaturalist, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=10)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=10)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(18490, 200)
        self.fc2 = nn.Linear(200, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 18490)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
lr = 1e-3
optimizer = optim.SGD(model.parameters(), lr=lr)
epochs = 2
log_interval = 10


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


for epoch in range(1, epochs + 1):
    train(epoch)
    # test()
