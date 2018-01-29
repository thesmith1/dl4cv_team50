import sys
import os
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
# print(sys.path)
from preprocessing.inaturalist_dataset import INaturalistDataset
import single_network.train_complete_set as train_script

# paths
data_dir = './data_preprocessed_224/'
annotations_dir = './annotations/'
train_annotations = '{}augmented_train2017.json'.format(annotations_dir)
val_annotations = '{}reduced_dataset_val2017.json'.format(annotations_dir)
test_annotations = '{}reduced_dataset_test2017.json'.format(annotations_dir)

# hyper-parameters
learning_rate = 1e-3
regularization_strength = 0
batch_size = 800
num_epochs = 10
loss = torch.nn.CrossEntropyLoss

# other parameters
do_testing = True
applied_transformations = transforms.Compose([transforms.ToTensor()])
non_printable = ["model", "optimizer", "loss"]

def setup_vgg19(parameters, output_categories=667):

    # get pre-trained model, change FC layer
    model = models.vgg19(pretrained=True)
    parameter_list = model.parameters()
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_categories))

    # move model to GPU
    if train_script.cuda:
        model = model.cuda()

    # create optimizer
    adam = optim.Adam(model.fc.parameters(), lr=parameters['lr'], weight_decay=parameters['reg'])
    return model, adam


if __name__ == '__main__':

    # setup
    params = {'lr':learning_rate, 'reg':regularization_strength}
    model, adam = setup_vgg19(params)

    # loading
    print("Loading training set...")
    inaturalist_train = INaturalistDataset(data_dir, train_annotations,
                                           transform=applied_transformations,
                                           modular_network_remap=False)
    output_categories = inaturalist_train.total_label_count
    train_loader = torch.utils.data.DataLoader(inaturalist_train, batch_size=batch_size, shuffle=True)

    print("\n\nLoading validation set...")
    inaturalist_val = INaturalistDataset(data_dir, val_annotations, transform=applied_transformations,
                                         modular_network_remap=False)
    val_loader = torch.utils.data.DataLoader(inaturalist_val, batch_size=batch_size, shuffle=True)

    if do_testing:
        print("\n\nLoading test set...")
        inaturalist_test = INaturalistDataset(data_dir, test_annotations,
                                              transform=applied_transformations,
                                              modular_network_remap=False)
        test_loader = torch.utils.data.DataLoader(inaturalist_test, batch_size, shuffle=True)
    else:
        inaturalist_test = None
        test_loader = None

    # create parameter set
    parameters = dict()
    parameters['model'] = model
    parameters['lr'] = learning_rate
    parameters['reg'] = regularization_strength
    parameters['batch-size'] = batch_size
    parameters['num-epochs'] = num_epochs
    parameters['optimizer'] = adam
    parameters['loss'] = loss
    parameters['output-filename'] = "{0}.pth".format(
        "_".join([str(key) + "=" + (parameter.__name__ if key in non_printable else str(parameter))
                  for key, parameter in parameters.items()]))

    # train with combination of hyper-parameters
    loaders = (train_loader, val_loader, test_loader)
    print("\n\nTraining model " + parameters['output-filename'])
    train_script.complete_train_validation(parameters, loaders, output_categories)