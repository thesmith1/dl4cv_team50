"""
This script executes the testing on the modular_network, loading all the best models
"""

import os
import sys

import torch
from torchvision import transforms

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from preprocessing.inaturalist_dataset import INaturalistDataset
from modular_network.modular_net import ModularNetwork

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

batch_size = 128
num_epochs = 1
start_lr = 1000
optimizers = ['sgd', 'adam', 'rmsprop']
loss_functions = ['cross_entropy', 'l1', 'nll', 'l2']

annotations_dir = './annotations/modular_network/'
test_annotations = '{}test2017_min.json'.format(annotations_dir)

transf = transforms.ToTensor()

print('Loading dataset for supercategories...')
inaturalist_test = INaturalistDataset(data_dir, test_annotations, transform=transf)
print('Dataset for supercategories loaded.')

test_loader = torch.utils.data.DataLoader(inaturalist_test, batch_size=batch_size, shuffle=True)

model = ModularNetwork({'train': None, 'val': None, 'test': inaturalist_test},
                       {'train': None, 'val': None, 'test': test_loader}, None, None,
                       cuda)
model_core = torch.load('./modular_network/models/path_to_core_net.pth')  # TODO
fc = None
model.load_model(fc, 'categories_net')
for cat in categories:
    model_species = torch.load('./modular_network/models/path_to_species_net.pth')  # TODO
    fc = None
    model.load_model(fc, cat)

model.test()
