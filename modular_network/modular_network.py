import sys, os

import copy
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from preprocessing.inaturalist_dataset import INaturalistDataset


def train_model(ds_train, ds_val, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, cuda_avail=False):
    datasets = {'train': ds_train, 'val': ds_val}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                # Iterate over data.
                for data in train_loader:
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if cuda_avail:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize
                    loss.backward()
                    optimizer.step()

                    # statistics
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    print('Running loss is ', running_loss)
            else:
                # Iterate over data.
                for data in val_loader:
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if cuda_avail:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # statistics
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    print('Running loss is ', running_loss)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects / len(datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def __main__():
    # Argument parsing
    parser = argparse.ArgumentParser(description='dl4cv_team50 Modular Network')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--num-classes', type=int, default=3, metavar='N',
                        help='number of classes (default: 3)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1000, metavar='LR',
                        help='initial learning rate (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', type=bool, default=False, metavar='SA',
                        help='if True, the program stores the best model (default: False)')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print('Starting script...')
    print('Checking cuda...')
    print('Cuda is ', args.cuda)

    data_dir = './data2/'
    annotations_dir = './annotations/'
    train_annotations = '{}train2017_min.json'.format(annotations_dir)
    val_annotations = '{}val2017_min.json'.format(annotations_dir)

    transf = transforms.ToTensor()

    print('Loading dataset...')
    inaturalist_train = INaturalistDataset(data_dir, train_annotations, transform=transf)
    inaturalist_val = INaturalistDataset(data_dir, val_annotations, transform=transf)
    print('Dataset loaded.')

    train_loader = torch.utils.data.DataLoader(inaturalist_train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(inaturalist_val, batch_size=args.batch_size, shuffle=True)

    print('Loading model...')
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_feat = model.fc.in_features
    model.fc = nn.Linear(num_feat, args.num_classes)
    print('Model loaded.')

    if args.cuda:
        model = model.cuda()

    loss_function = nn.CrossEntropyLoss()
    # Alternatives for loss function are:
    # L1, MSELoss (L2), NLLLoss

    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    # Alternatives for optim are:
    # Adam, Adagrad, Adadelta, RMSprop

    # Scheduler changes every step_size epochs the learning rate by a factor of gamma
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print('Starting training...')
    model = train_model(inaturalist_train, inaturalist_val, model, train_loader, val_loader, loss_function, optimizer,
                        exp_lr_scheduler, num_epochs=args.epochs, cuda_avail=args.cuda)
    print('Model trained.')

    if args.save:
        print('Saving best model...')
        torch.save(model, './mod1.pth')
        print('Best model saved.')


__main__()
