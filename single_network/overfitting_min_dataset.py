#!/usr/bin/env python3
"""
script for overfitting small dataset (see annotations/*_min.json)
"""
import sys, os
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

# print(sys.path)

from preprocessing.inaturalist_dataset import INaturalistDataset
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torchvision import models, transforms
from torch.autograd import Variable

cuda = False # torch.cuda.is_available()

# parameters
batch_size = 10
lr = 1e-3
epochs = 10
log_interval = 10
loss = nn.CrossEntropyLoss()
output_categories = 3
optimizer = optim.Adam

# set directories
data_dir = './data_preprocessed/'
annotations_dir = './annotations/single_network/'
train_annotations = '{}train2017.json'.format(annotations_dir)
val_annotations = '{}val2017.json'.format(annotations_dir)

# create data sets
applied_transformation = transforms.Compose([transforms.ToTensor()])
inaturalist_train = INaturalistDataset(data_dir, train_annotations, transform=applied_transformation)
inaturalist_val = INaturalistDataset(data_dir, val_annotations, transform=applied_transformation)

# create loaders for the data sets
train_loader = torch.utils.data.DataLoader(inaturalist_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(inaturalist_val, batch_size=batch_size, shuffle=True)

# get pre-trained model, change FC layer
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
fc_in_features = model.fc.in_features
model.fc = nn.Linear(fc_in_features, output_categories)

# create optimizer
optimizer = optimizer(model.fc.parameters(), lr=lr)

# move model to GPU
if cuda:
    model = model.cuda()


# single epoch of training method
def train(epoch):

    # set train mode
    model.train()

    # for each batch
    for batch_idx, (data, targets) in enumerate(train_loader):

        # initialization
        _, target = targets
        data, (target) = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        # forward pass
        output = model(data)

        # loss
        loss_value = loss(output, target)

        # backward pass
        loss_value.backward()

        # weight upgrade
        optimizer.step()

        # log
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss_value.data[0]))


def evaluate(dataset_loader):

    # set evaluation mode
    model.eval()

    # initialization
    val_loss_value = 0
    correct = 0

    # for each batch
    for data, targets in dataset_loader:

        # initialization
        _, target = targets

        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # forward pass
        output = model(data)

        # compute loss
        val_loss_value += loss(output, target).data[0]  # sum up batch loss

        # predict
        predicted_labels = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += predicted_labels.eq(target.data.view_as(predicted_labels)).cpu().sum()

    val_loss_value /= len(val_loader.dataset)

    # log
    print('Evaluation results:\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss_value, correct, len(dataset_loader.dataset),
        100. * correct / len(dataset_loader.dataset)))


if __name__ == '__main__':

    # train
    for epoch_count in range(1, epochs + 1):
        train(epoch_count)

    # evaluation on train set
    print("\n\nEvaluating model on training set...")
    evaluate(train_loader)

    # evaluation on validation set
    print("Evaluating model on validation set...")
    evaluate(val_loader)
