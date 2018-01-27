#!/usr/bin/env python3
"""
script for loading a model trained on the iNaturalist dataset back (not just a pre-trained one)
"""
import sys
import os
import torch
from torchvision import transforms

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
# print(sys.path)

from preprocessing.inaturalist_dataset import INaturalistDataset
from single_network.train_complete_set import evaluate

# inputs
models_base_folder = './single_network/models/'
pth_filename = 'reg=0.0001_optimizer=Adam_num-epochs=1_loss=CrossEntropyLoss_model=resnet50_lr=0.0005_batch-size=800' \
               '.pth '
test_annotations = './annotations/reduced_dataset_test2017.json'
test_dir = lambda input_size: './data_preprocessed_{}/'.format(input_size)
test_batch_size = 100
applied_transformations = transforms.Compose([transforms.ToTensor()])


def load_model(filename):

    # accepted parameters
    accepted_parameters = ['model', 'lr', 'reg', 'batch-size', 'num-epochs', 'optimizer', 'loss']

    # split parameters
    parameter_couples = filename[:-4].split("_")

    model_parameters = dict()
    for parameter_couple in parameter_couples:
        parameter_name, parameter_value = parameter_couple.split("=")
        if parameter_name not in accepted_parameters:
            raise ValueError("Invalid parameter name '%s'" % parameter_name)
        else:
            model_parameters[parameter_name] = parameter_value

    return torch.load(open(models_base_folder + filename, "rb")), model_parameters


if __name__ == '__main__':

    print("Loading model...")
    model, parameters = load_model(pth_filename)
    model.eval()
    print("done.")

    # print(type(model))

    # produce test loader
    input_size = 224
    inaturalist_test = INaturalistDataset(test_dir(input_size), test_annotations,
                                          transform=applied_transformations,
                                          modular_network_remap=False)
    test_loader = torch.utils.data.DataLoader(inaturalist_test, batch_size=test_batch_size)

    # find loss
    if parameters['loss'] == 'CrossEntropyLoss':
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss '%s'" % parameters['loss'])

    evaluate(model, loss, test_loader)
