"""
This script launches the training on the core of the modular network,
the one which classifies the supercategories;
at the end of the process, it will save the results and the best model
"""

import os
import sys
import pickle
import argparse

import torch
from torchvision import transforms

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
ext_lib_path = os.path.abspath(os.path.join(__file__, '../../preprocessing'))
sys.path.append(ext_lib_path)

from inaturalist_dataset import INaturalistDataset
from modular_net import ModularNetwork


parser = argparse.ArgumentParser(description='dl4cv_team50 Modular Network')
parser.add_argument('--model', default=None, metavar='m', dest='model',
                    help='path to the model to be loaded')
parser.add_argument('--save', type=bool, default=True, metavar='s', dest='save',
                    help='whether to save the best model or not')
parser.add_argument('--lr', type=float, default=1e-3, metavar='l', dest='lr',
                    help='inital learning rate')
parser.add_argument('--gamma', type=float, default=0.1, metavar='g', dest='gamma',
                    help='gamma for the lr scheduler')
parser.add_argument('--step-size', type=int, default=1, metavar='t', dest='step_size',
                    help='step size for the lr scheduler')
parser.add_argument('--batch-size', type=int, default=850, metavar='b', dest='batch_size',
                    help='batch size for training')
parser.add_argument('--epochs', type=int, default=1, metavar='e', dest='epochs',
                    help='number of total epochs')
parser.add_argument('--optimizers', default=None, nargs='+', metavar='o', dest='optimizers',
                    help='list of optimizers to be used')
parser.add_argument('--loss-functions', default=None, nargs='+', metavar='f', dest='loss_functions',
                    help='list of loss functions to be used')
parser.add_argument('--weight-decay', type=int, default=0, metavar='w', dest='weight_decay',
                    help='regularization factor')
args = parser.parse_args()

cuda = torch.cuda.is_available()

torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)
print('Starting script...')
print('Checking is_cuda...')
print('Cuda is', cuda)

# categories = ['Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista',
#                'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia']
categories = ['Amphibia', 'Animalia', 'Mammalia', 'Reptilia']
num_species = {'Actinopterygii': 53, 'Amphibia': 115, 'Animalia': 77, 'Arachnida': 56,
                'Aves': 964, 'Chromista': 9, 'Fungi': 121, 'Insecta': 1021, 'Mammalia': 186,
                'Mollusca': 93, 'Plantae': 2101, 'Protozoa': 4, 'Reptilia': 289}

data_dir = './data_preprocessed/'

batch_size = args.batch_size
num_epochs = args.epochs
# optimizers = ['sgd', 'adam', 'rmsprop']
# loss_functions = ['cross_entropy', 'l1', 'nll', 'l2']
start_lr = args.lr
step_size = args.step_size
gamma = args.gamma
weight_decay = args.weight_decay
optimizers = args.optimizers
loss_functions = args.loss_functions

annotations_dir = './annotations/'
train_annotations = '{}reduced_dataset_train2017.json'.format(annotations_dir)
val_annotations = '{}reduced_dataset_val2017.json'.format(annotations_dir)

transf = transforms.ToTensor()

print('Loading dataset for supercategories...')
inaturalist_train = INaturalistDataset(data_dir, train_annotations, transform=transf)
inaturalist_val = INaturalistDataset(data_dir, val_annotations, transform=transf)
print('Dataset for supercategories loaded.')

train_loader = torch.utils.data.DataLoader(inaturalist_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(inaturalist_val, batch_size=batch_size, shuffle=True)

for optimizer in optimizers:
    for loss in loss_functions:
        train_params = {'optimizer': optimizer, 'learning_rate': start_lr, 'gamma': gamma, 'step_size': step_size,
                        'weight_decay': weight_decay}

        if args.model is None:
            model = ModularNetwork({'train': inaturalist_train, 'val': inaturalist_val, 'test': None},
                                   {'train': train_loader, 'val': val_loader, 'test': None}, train_params, loss,
                                   cuda)
        else:
            print('Loading model from file...')
            model = torch.load(args.model)
            model.set_parameters({'train': inaturalist_train, 'val': inaturalist_val, 'test': None},
                                   {'train': train_loader, 'val': val_loader, 'test': None}, train_params, loss,
                                   cuda)
            print('Model loaded.')

        best_model, hist_acc, hist_loss = model.train('categories_net', num_epochs)
        if args.save:
            print('Saving best model...')
            model_filename = './modular_network/models/resnet50_{}_model_{}_{}.pth'.format('supercategories',
                                                                                           optimizer, loss)
            torch.save(model, model_filename)
            print('Best model saved.')
            print('Saving results...')
            if args.model is not None:
                subfolders = args.model.split('.')[1].split('/')
                old_results_filename = './' + subfolders[1] + '/results/' + subfolders[3] + '.pkl'
                old_results = pickle.load(open(old_results_filename, 'rb'))
                old_hist_acc = old_results['accuracy']
                old_hist_loss = old_results['loss']
                new_hist_acc = {'train': old_hist_acc['train'] + hist_acc['train'],
                                'val': old_hist_acc['val'] + hist_acc['val']}
                new_hist_loss = {'train': old_hist_loss['train'] + hist_loss['train'],
                                'val': old_hist_loss['val'] + hist_loss['val']}
            else:
                new_hist_loss = hist_loss
                new_hist_acc = hist_acc
            results = {'accuracy': new_hist_acc, 'loss': new_hist_loss}
            results_filename = './modular_network/results/resnet50_{}_results_{}_{}.pkl'.format('supercategories',
                                                                                                optimizer, loss)
            with open(results_filename, 'wb') as output:
                pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
            print('Results saved.')
