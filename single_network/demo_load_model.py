#!/usr/bin/env python3
"""
script for loading a model trained on the iNaturalist dataset back (not just a pre-trained one)
"""
import sys
import os
import torch

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
# print(sys.path)


models_base_folder = './annotations/models/'
pth_filename = 'model=resnet50_lr=0.001_reg=0_batch-size=16_num-epochs=1_optimizer=Adam_loss=CrossEntropyLoss.pth'
complete_pth = models_base_folder + pth_filename


def load_model(pth_file):

    # accepted parameters
    accepted_parameters = ['model', 'lr', 'reg', 'batch-size', 'num-epches', 'optimizer', 'loss']

    # split parameters
    parameter_couples = pth_file[:-4].split("_")

    model_parameters = dict()
    for parameter_couple in parameter_couples:
        parameter_name, parameter_value = parameter_couple.split("=")
        if parameter_name not in accepted_parameters:
            raise ValueError("Invalid parameter name '%s'" % parameter_name)
        else:
            model_parameters[parameter_name] = parameter_value

    return torch.load(open(pth_file, "r")), model_parameters


if __name__ == '__main__':
    model, parameters = load_model(complete_pth)
    model.eval()

    print(model.__name__)
    # TODO: continue




