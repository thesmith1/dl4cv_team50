#!/usr/bin/env python3
"""
script for overfitting small dataset (see annotations/*_min.json)
"""
import sys, os, numpy
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

# print(sys.path)

from preprocessing.inaturalist_dataset import INaturalistDataset
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torchvision import models, transforms
from torch.autograd import Variable
import copy

cuda = torch.cuda.is_available()

# parameters
batch_size = 800
lr = 1e-3
epochs = 1
log_interval = 1
loss = nn.CrossEntropyLoss()
output_categories = 5089
chosen_optimizer = optim.Adam
chosen_model = models.resnet50

# set directories
data_dir = './data_preprocessed_299/'
annotations_dir = './annotations/'
train_annotations = '{}train2017_min.json'.format(annotations_dir)
val_annotations = '{}val2017min.json'.format(annotations_dir)
#test_annotations = '{}test2017_new.json'.format(annotations_dir)
applied_transformations = transforms.Compose([transforms.ToTensor()])


# get pre-trained model, change FC layer
model = chosen_model(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
fc_in_features = model.fc.in_features
model.fc = nn.Linear(fc_in_features, output_categories)

# move model to GPU
if cuda:
    model = model.cuda()

# create optimizer
optimizer = chosen_optimizer(model.fc.parameters(), lr=lr)


# single epoch of training method
def train(epoch):

    # set train mode
    model.train()

    # for each batch
    for batch_idx, (data, targets) in enumerate(train_loader):

        # keep only species target
        _, target = targets
        # print("Batch dim:", data.shape)
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
            print('Train Epoch: {} [{}/{} ({:.2f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx * len(data) / len(train_loader.dataset),
                loss_value.data[0]), end='\r')


def evaluate(dataset_loader):

    # set evaluation mode
    model.eval()

    # initialization
    loss_value = 0
    correct = 0

    # for each batch
    for batch_idx, (data, targets) in enumerate(dataset_loader):

        # keep only species target
        _, target = targets

        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # forward pass
        output = model(data)

        # compute loss
        loss_value += loss(output, target).data[0]  # sum up batch loss

        # predict
        predicted_labels = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += predicted_labels.eq(target.data.view_as(predicted_labels)).cpu().sum()

        # log
        if batch_idx % log_interval == 0:
            print('Evaluated images: {}/{} ({:.2f}%)'.format(batch_idx * len(data),
                                                            len(dataset_loader.dataset),
                                                            100. * batch_idx * len(data)/ len(dataset_loader.dataset)),
                  end='\r')

    loss_value /= len(val_loader.dataset)

    # final log
    print('\nEvaluation results:\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss_value, correct, len(dataset_loader.dataset),
        100. * correct / len(dataset_loader.dataset)))


if __name__ == '__main__':

    # training
    print("\n\nLoading training set...")
    inaturalist_train = INaturalistDataset(data_dir, train_annotations, transform=None,
                                           modular_network_remap=False)
    labels = numpy.zeros(len(inaturalist_train))    
    stacked = numpy.zeros((len(inaturalist_train),268203))
    for i, photo_data in enumerate(inaturalist_train):
        #print(numpy.shape(numpy.array(photo_data[0]).flatten()))
        stacked[i] = numpy.array(photo_data[0]).flatten()
        labels[i] = photo_data[1][1]
        #for component in (photo_data[1]):
        #    print("Class: ", component)
    print(numpy.shape(stacked))
    print(stacked[0])
    print("Labels: ", labels)
    """
    train_loader = torch.utils.data.DataLoader(inaturalist_train, batch_size=batch_size, shuffle=True)
    print("Starting training (%d epochs)" % epochs)
    for epoch_count in range(1, epochs + 1):
        train(epoch_count)

    # evaluation on validation set
    print("\n\nLoading validation set...")
    inaturalist_val = INaturalistDataset(data_dir, val_annotations, transform=applied_transformations,
                                         modular_network_remap=False)
    val_loader = torch.utils.data.DataLoader(inaturalist_val, batch_size=batch_size, shuffle=True)
    print("Evaluating model on validation set...")
    evaluate(val_loader)

    # evaluation on test set
    print("\n\nLoading test set...")
    inaturalist_test = INaturalistDataset(data_dir, test_annotations, transform=applied_transformations,
                                          modular_network_remap=False)
    test_loader = torch.utils.data.DataLoader(inaturalist_test, batch_size, shuffle=True)
    print("Evaluating model on test set...")
    # evaluate(test_loader)

    model_dict = copy.copy(model.state_dict())
    torch.save(model_dict, "model_single_epoch.pth")
    """

