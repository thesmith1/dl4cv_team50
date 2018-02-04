#!/usr/bin/env python3
"""
script for combining trained models.
"""
import sys
import os
import torch
from torchvision import transforms

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
# print(sys.path)

from preprocessing.inaturalist_dataset import INaturalistDataset
from single_network.train_complete_set import *

# inputs
models_base_folder = './single_network/models/'
pth_filenames = ['loss=CrossEntropyLoss_model=resnet152_lr=0.0001_optimizer=Adam_batch-size=80_reg=1e-05_num-epochs=5.pth', 'loss=CrossEntropyLoss_model=resnet152_lr=0.0001_optimizer=Adam_batch-size=80_reg=0.0001_num-epochs=5.pth']
test_annotations = './annotations/reduced_dataset_test2017.json'
test_dir = lambda input_size: './data_preprocessed_{}/'.format(input_size)
test_batch_size = 100
applied_transformations = transforms.Compose([transforms.ToTensor()])

def evaluate(models, loss, dataset_loader):
    # set evaluation mode
    for model in models:
        model.eval()

    # initialization
    loss_value = 0
    top1_correct = 0
    top5_correct = 0

    # for each batch
    for batch_idx, (data, targets) in enumerate(dataset_loader):

        # keep only species target
        _, target = targets

        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # forward pass
        output = models[0](data)
        for model in models[1:]:
            output = output + model(data)

        # compute loss
        loss_value += loss(output, target).data[0]  # sum up batch loss

        # predict
        # top1_predicted_labels = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct1, correct2 = correct_predictions(output, target)

        # update correct
        top1_correct += correct1  # top1_predicted_labels.eq(target.data.view_as(top1_predicted_labels)).cpu().sum()
        top5_correct += correct2

        # log
        if batch_idx % log_interval == 0:
            print('Evaluated images: {}/{} ({:.2f}%)'.format(batch_idx * dataset_loader.batch_size,
                                                             len(dataset_loader.dataset),
                                                             100. * batch_idx / len(dataset_loader)), end='\r')

    loss_value /= len(dataset_loader.dataset)

    # final log
    print('Evaluation completed.\nEvaluation results:\n'
          'Average loss: {:.4f},\nTOP 1 Accuracy: {}/{} ({:.2f}%),\nTOP 5 Accuracy: {}/{} ({:.2f}%)\n'
          .format(loss_value,
                  top1_correct, len(dataset_loader.dataset), 100. * top1_correct / len(dataset_loader.dataset),
                  top5_correct, len(dataset_loader.dataset), 100. * top5_correct / len(dataset_loader.dataset)))

    return loss_value, top1_correct, top5_correct, len(dataset_loader.dataset)



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

    print("Loading models...")
    models = []
    for path in pth_filenames:
        model, parameters = load_model(path)
        models.append(model)
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

    evaluate(models, loss, test_loader)
