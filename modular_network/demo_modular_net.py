import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from preprocessing.inaturalist_dataset import INaturalistDataset
from modular_network.modular_net import ModularNetwork

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
print('Cuda is', args.cuda)

data_dir = './data2/'
annotations_dir = './annotations/modular_network/Mammalia/'
train_annotations = '{}train2017_min.json'.format(annotations_dir)
val_annotations = '{}val2017_min.json'.format(annotations_dir)

transf = transforms.ToTensor()

print('Loading dataset...')
inaturalist_train = INaturalistDataset(data_dir, train_annotations, transform=transf)
inaturalist_val = INaturalistDataset(data_dir, val_annotations, transform=transf)
print('Dataset loaded.')

train_loader = torch.utils.data.DataLoader(inaturalist_train, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(inaturalist_val, batch_size=args.batch_size, shuffle=True)

train_params = {'optimizer': 'sgd', 'learning_rate': args.lr}
loss_function = 'cross_entropy'

model = ModularNetwork({'train': inaturalist_train, 'val': inaturalist_val, 'test': inaturalist_val},
                       {'train': train_loader, 'val': val_loader, 'test': val_loader}, train_params, loss_function,
                       args.cuda)

best_model, hist_acc, hist_loss = model.train('Mammalia', args.epochs)

print(hist_acc, hist_loss)
'''
x_axis = np.arange(0, args.epochs, 1)
figure, ax = plt.subplots()
ax.plot(x_axis, hist_acc['train'])
ax.set(xlabel='Epochs', ylabel='Training accuracy')
ax.grid()
plt.show()
'''
if args.save:
    print('Saving best model...')
    torch.save(model, './mod1.pth')
    print('Best model saved.')

model.test()
