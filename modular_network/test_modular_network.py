"""
This script executes the testing on the modular_network, loading all the best models
"""

import os
import sys
import argparse

import torch
from torchvision import transforms

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from inaturalist_dataset import INaturalistDataset
from modular_net import ModularNetwork

parser = argparse.ArgumentParser(description='dl4cv_team50 Modular Network')
parser.add_argument('--batch-size', type=int, default=850, metavar='b', dest='batch_size',
                    help='batch size for training')
args = parser.parse_args()

cuda = torch.cuda.is_available()

torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)
print('Starting script...')
print('Checking cuda...')
print('Cuda is', cuda)

# categories = ['Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista',
#                'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia']
categories = ['Amphibia', 'Animalia', 'Mammalia', 'Reptilia']
num_species = {'Actinopterygii': 53, 'Amphibia': 115, 'Animalia': 77, 'Arachnida': 56,
                'Aves': 964, 'Chromista': 9, 'Fungi': 121, 'Insecta': 1021, 'Mammalia': 186,
                'Mollusca': 93, 'Plantae': 2101, 'Protozoa': 4, 'Reptilia': 289}
models = {'categories_net': './modular_network/models/resnet50_supercategories_results_adam_cross_entropy_9_84.pth',
          'Amphibia': './modular_network/models/resnet50_Amphibia_results_adam_cross_entropy_2_44.pth',
          'Animalia': './modular_network/models/resnet50_Animalia_results_adam_cross_entropy_1_80.pth',
          'Mammalia': './modular_network/models/resnet50_Mammalia_results_adam_cross_entropy_1_57.pth',
          'Reptilia': './modular_network/models/resnet50_Reptilia_results_adam_cross_entropy_2_41.pth'}

data_dir = './data_preprocessed/'

batch_size = args.batch_size

annotations_dir = './annotations/modular_network/'
test_annotations = '{}reduced_dataset_test2017.json'.format(annotations_dir)

transf = transforms.ToTensor()

print('Loading dataset...')
inaturalist_test = INaturalistDataset(data_dir, test_annotations, transform=transf)
print('Dataset loaded.')

test_loader = torch.utils.data.DataLoader(inaturalist_test, batch_size=batch_size, shuffle=True)

print('Loading the models to be tested...')
model = ModularNetwork({'train': None, 'val': None, 'test': inaturalist_test},
                       {'train': None, 'val': None, 'test': test_loader}, None, None,
                       cuda)
model_core = torch.load(models['categories_net'])
fc = model_core.categories_model_fc = model_core
model.load_model(fc, 'categories_net')
for cat in categories:
    model_species = torch.load(models[cat])
    fc = model_species.mini_net_model[cat]
    model.load_model(fc, cat)
print('Models loaded.')

model.test()
