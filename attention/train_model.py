#!/usr/bin/env python3
"""
script for overfitting small dataset (see annotations/*_min.json)
"""
import sys, os, copy
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

# print(sys.path)

from preprocessing.inaturalist_dataset import INaturalistDataset
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torchvision import models, transforms
from torch.autograd import Variable

from attention_network import *
from classification_network import *
from feature_extraction import extract_features

cuda = torch.cuda.is_available()

# parameters
batch_size = 10
lr = 1e-3
epochs = 1
log_interval = 10
loss = nn.CrossEntropyLoss()
output_categories = 3
optimizer = optim.Adam

# set directories
data_dir = './data_min_preprocessed_224/'
annotations_dir = './annotations/modular_network/Mammalia/'
train_annotations = '{}train2017_min.json'.format(annotations_dir)
val_annotations = '{}val2017_min.json'.format(annotations_dir)

# create data sets
applied_transformation = transforms.Compose([transforms.ToTensor()])
inaturalist_train = INaturalistDataset(data_dir, train_annotations, transform=applied_transformation,
                                       modular_network_remap=False)
inaturalist_val = INaturalistDataset(data_dir, val_annotations, transform=applied_transformation,
                                     modular_network_remap=False)

# create loaders for the data sets
train_loader = torch.utils.data.DataLoader(inaturalist_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(inaturalist_val, batch_size=batch_size, shuffle=True)

# load full image resnet
# PLACEHOLDER #
fullres = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
fc_in_features = fullres.fc.in_features
fullres.fc = nn.Linear(fc_in_features, output_categories)
# PLACEHOLDER #

# set up resnets
clas1 = ClassificationCNN()
clas2 = ClassificationCNN()

#setup attention
att1 = AttentionCNN()
att2 = AttentionCNN()

# create optimizers
opt1 = optimizer(clas1.parameters(), lr=lr)
opt2 = optimizer(clas2.parameters(), lr=lr)

# move model to GPU
if cuda:
    clas1 = clas1.cuda()
    clas2 = clas2.cuda()
    att1 = att1.cuda()
    att2 = att2.cuda()

# single epoch of training method
def train(epoch):

    # set train mode
    clas1.train()
    clas2.train()
    att1.train()
    att2.train()

    # for each batch
    for batch_idx, (data, targets) in enumerate(train_loader):

        # initialization
        _, target = targets
        # print(data.shape)
        data, (target) = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        opt1.zero_grad()
        opt2.zero_grad()

        # forward pass
        features = extract_features(fullres, data)
        attention_map1 = att1(features)
        attention_map2 = att2(features)

        (region1, region1_coord) = crop_region(features, attention_map1)
        (region2, region2_coord) = crop_region(features, attention_map2)

        out1 = clas1(region1)
        out2 = (clas2(region2) + out1)/2

        # loss
        loss1 = loss(out1, target)
        loss2 = loss(out2, target)
        loss_value = (loss1 + loss2)/2
        reward_loss = ()

        # backward pass
        loss_value.backward()
        reward_grad = ()

        # weight upgrade
        opt1.step()
        opt2.step()
        reward_opt()

        # log
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss_value.data[0]))


def evaluate(dataset_loader):

    # set evaluation mode
    clas1.eval()
    clas2.eval()
    att1.eval()
    att2.eval()

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
        features = extract_features(fullres, data)
        attention_map1 = att1(features)
        attention_map2 = att2(features)

        (region1, region1_coord) = crop_region(data, attention_map1)
        (region2, region2_coord) = crop_region(data, attention_map2)

        out1 = clas1(region1)
        out2 = (clas2(region2) + out1)/2

        # compute loss
        val_loss_value += loss(out2, target).data[0]  # sum up batch loss

        # predict
        predicted_labels = out2.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
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

    clas1_dict = copy.copy(clas1.state_dict())
    torch.save(clas1_dict, "clas1_overfit.pth")
    clas2_dict = copy.copy(clas2.state_dict())
    torch.save(clas2_dict, "clas2_overfit.pth")
    att1_dict = copy.copy(att1.state_dict())
    torch.save(att1_dict, "att1_overfit.pth")
    att2_dict = copy.copy(att2.state_dict())
    torch.save(att2_dict, "att2_overfit.pth")
