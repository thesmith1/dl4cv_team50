import sys
import os
from torchvision import models, transforms
import torch
import torch.optim as optim

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
# print(sys.path)
from single_network.train_complete_set import complete_train_validation
from preprocessing.inaturalist_dataset import INaturalistDataset

# models
architectures = [models.resnet50, models.inception_v3]
model_input_sizes = {models.resnet50: 224, models.inception_v3: 299}

# hyper-parameters
learning_rates = [1e-4, 1e-3, 1e-2]
regularization_strengths = [0, 1e-4, 1e-3, 1e-2, 1e-1]
batch_size = 800
num_epochs = 1
optimizer = optim.Adam
loss = torch.nn.CrossEntropyLoss
applied_transformations = transforms.Compose([transforms.ToTensor()])

# set directories
data_dir = lambda size: './data_preprocessed_{}/'.format(size)
annotations_dir = './annotations/'
train_annotations = '{}augmented_train2017.json'.format(annotations_dir)
val_annotations = '{}reduced_dataset_val2017.json'.format(annotations_dir)
test_annotations = '{}reduced_dataset_test2017.json'.format(annotations_dir)

# other parameters
do_testing = False
non_printable = ["model", "optimizer", "loss"]

if __name__ == '__main__':

    for model in architectures:

        input_size = model_input_sizes[model]

        # loading
        print("Loading training set...")
        inaturalist_train = INaturalistDataset(data_dir(input_size), train_annotations,
                                               transform=applied_transformations,
                                               modular_network_remap=False)
        output_categories = inaturalist_train.total_label_count
        train_loader = torch.utils.data.DataLoader(inaturalist_train, batch_size=batch_size, shuffle=True)

        print("\n\nLoading validation set...")
        inaturalist_val = INaturalistDataset(data_dir(input_size), val_annotations, transform=applied_transformations,
                                             modular_network_remap=False)
        val_loader = torch.utils.data.DataLoader(inaturalist_val, batch_size=batch_size, shuffle=True)

        if do_testing:
            print("\n\nLoading test set...")
            inaturalist_test = INaturalistDataset(data_dir(input_size), test_annotations,
                                                  transform=applied_transformations,
                                                  modular_network_remap=False)
            test_loader = torch.utils.data.DataLoader(inaturalist_test, batch_size, shuffle=True)
        else:
            inaturalist_test = None
            test_loader = None

        for lr in learning_rates:
            for reg in regularization_strengths:

                # create parameter set
                parameters = dict()
                parameters['model'] = model
                parameters['lr'] = lr
                parameters['reg'] = reg
                parameters['batch-size'] = batch_size
                parameters['num-epochs'] = num_epochs
                parameters['optimizer'] = optimizer
                parameters['loss'] = loss
                parameters['output-filename'] = "{0}.pth".format(
                    "_".join([str(key) + "=" + (parameter.__name__ if key in non_printable else str(parameter))
                              for key, parameter in parameters.items()]))

                # train with combination of hyper-parameters
                loaders = (train_loader, val_loader, test_loader)
                print("\n\nTraining model " + parameters['output-filename'])
                complete_train_validation(parameters, loaders, output_categories, validation_during_training=True)
