import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from torch.optim import lr_scheduler


class ModularNetwork(object):
    def __init__(self, datasets, train_loader, val_loader, train_params, loss_function, cuda_avail=False):
        self.categories = ['Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista',
                           'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia']
        self.num_species = {'Actinopterygii': 53, 'Amphibia': 115, 'Animalia': 77, 'Arachnida': 56,
                            'Aves': 964, 'Chromista': 9, 'Fungi': 121, 'Insecta': 1021, 'Mammalia': 186,
                            'Mollusca': 93, 'Plantae': 2101, 'Protozoa': 4, 'Reptilia': 289}
        self.num_classes = len(self.categories)
        self.datasets = datasets
        self.loaders = {'train': train_loader, 'val': val_loader}
        self.optimizer = train_params['optimizer']
        self.learning_rate = train_params['learning_rate']
        self.loss_function = loss_function
        self.cuda = cuda_avail
        # The big network which classifies categories
        print('Loading the network for categories...')
        self.feat_model = models.resnet50(pretrained=True)
        for param in self.feat_model.parameters():
            param.requires_grad = False
        num_feat = self.feat_model.fc.in_features
        self.categories_model_fc = nn.Linear(num_feat, self.num_classes)
        print('Done.')
        # Create the smaller networks, one for each category
        print('Loading the smaller networks for the species...')
        self.mini_net_model = {}
        self.num_species = {}
        for cat in self.categories:
            self.num_species[cat] = 3  # TODO: delete this line to make it work with entire dataset
            self.mini_net_model[cat] = nn.Linear(num_feat, self.num_species[cat])
        print('Done.')

    def train(self, what, num_epochs):
        since = time.time()

        # Build the network to be trained
        print('Building the model to be trained...')
        if what == 'categories_net':
            model = self.feat_model
            model.fc = self.categories_model_fc
        else:
            model = self.feat_model
            model.fc = self.mini_net_model[what]
        print('Done.')

        if self.optimizer == 'sgd':
            optimizer = optim.SGD(model.fc.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = None  # TODO: missing implementation of other optimizers
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_models = {what: copy.deepcopy(model.state_dict())}
        best_acc = 0.0

        print('Starting training...')
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)
                else:
                    model.train(False)

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for data in self.loaders[phase]:
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if self.cuda:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = self.loss_function(outputs, labels)

                    if phase == 'train':
                        # backward + optimize
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    print('Running loss is ', running_loss)

                epoch_loss = running_loss / len(self.datasets[phase])
                epoch_acc = running_corrects / len(self.datasets[phase])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_models[what] = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_models[what])
        return model

    def test(self):
        pass