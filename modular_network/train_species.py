"""
This script launches the training on all the branches of the modular network,
the ones which classify the species inside the supercategory;
at the end of the process, it will save the results and the best models
"""

import os
import sys
import pickle

import torch
from torchvision import transforms

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
ext_lib_path = os.path.abspath(os.path.join(__file__, '../../preprocessing'))
sys.path.append(ext_lib_path)

from inaturalist_dataset import INaturalistDataset
from modular_net import ModularNetwork

cuda = torch.cuda.is_available()
save = True

torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)
print('Starting script...')
print('Checking cuda...')
print('Cuda is', cuda)

categories = ['Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista',
               'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia']
num_species = {'Actinopterygii': 53, 'Amphibia': 115, 'Animalia': 77, 'Arachnida': 56,
                'Aves': 964, 'Chromista': 9, 'Fungi': 121, 'Insecta': 1021, 'Mammalia': 186,
                'Mollusca': 93, 'Plantae': 2101, 'Protozoa': 4, 'Reptilia': 289}

data_dir = './data_preprocessed/'

batch_size = 800
num_epochs = 1
start_lr = 1e-3
optimizers = ['sgd']
loss_functions = ['cross_entropy']
# start_lr = 1000
# optimizers = ['sgd', 'adam', 'rmsprop']
# loss_functions = ['cross_entropy', 'l1', 'nll', 'l2']

for cat in categories:
    annotations_dir = './annotations/modular_network/{}/'.format(cat)
    train_annotations = '{}train2017_min.json'.format(annotations_dir, cat)
    val_annotations = '{}val2017_min.json'.format(annotations_dir, cat)

    transf = transforms.ToTensor()

    print('Loading dataset for supercategories...')
    inaturalist_train = INaturalistDataset(data_dir, train_annotations, transform=transf)
    inaturalist_val = INaturalistDataset(data_dir, val_annotations, transform=transf)
    print('Dataset for supercategories loaded.')

    train_loader = torch.utils.data.DataLoader(inaturalist_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(inaturalist_val, batch_size=batch_size, shuffle=True)

    for optimizer in optimizers:
        for loss in loss_functions:
            train_params = {'optimizer': optimizer, 'learning_rate': start_lr}

            model = ModularNetwork({'train': inaturalist_train, 'val': inaturalist_val, 'test': None},
                                   {'train': train_loader, 'val': val_loader, 'test': None}, train_params, loss,
                                   cuda)

            best_model, hist_acc, hist_loss = model.train(cat, num_epochs)
            if save:
                print('Saving best model...')
                model_filename = './modular_network/models/resnet50_{}_model_{}_{}.pth'.format(cat, optimizer, loss)
                torch.save(model, model_filename)
                print('Best model saved.')
                print('Saving results...')
                results = {'accuracy': hist_acc, 'loss': hist_loss}
                results_filename = './modular_network/results/resnet50_{}_results_{}_{}.pkl'.format(cat, optimizer,
                                                                                                    loss)
                with open(results_filename, 'wb') as output:
                    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
                print('Results saved.')
