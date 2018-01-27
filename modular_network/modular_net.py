import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from torch.optim import lr_scheduler


class ModularNetwork(object):
    def __init__(self, datasets, loaders, train_params, loss_function, cuda_avail=False):
        # self.categories = ['Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista',
        #                    'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia']
        self.categories = ['Amphibia', 'Animalia', 'Mammalia', 'Reptilia']
        self.num_species = {'Actinopterygii': 53, 'Amphibia': 115, 'Animalia': 77, 'Arachnida': 56,
                            'Aves': 964, 'Chromista': 9, 'Fungi': 121, 'Insecta': 1021, 'Mammalia': 186,
                            'Mollusca': 93, 'Plantae': 2101, 'Protozoa': 4, 'Reptilia': 289}
        self.num_classes = len(self.categories)
        self.datasets = datasets
        self.loaders = loaders
        self.optimizer = train_params['optimizer']
        self.weight_decay = train_params['weight_decay']
        self.learning_rate = train_params['learning_rate']
        self.gamma = train_params['gamma']
        self.step_size = train_params['step_size']
        if loss_function == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function == 'l1':
            self.loss_function = nn.L1Loss()
        elif loss_function == 'nll':
            self.loss_function = nn.NLLLoss()
        elif loss_function == 'l2':
            self.loss_function = nn.MSELoss()
        else:
            raise AttributeError('Invalid choice of loss function')
        self.cuda = cuda_avail
        # The big network which classifies categories
        print('Loading the network for categories...')
        self.feat_model = models.resnet50(pretrained=True)
        for param in self.feat_model.parameters():
            param.requires_grad = False
        num_feat = self.feat_model.fc.in_features
        self.hidden_layer_size = int(np.sqrt(num_feat))
        self.categories_model_fc = nn.Sequential(
            nn.Linear(num_feat, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.num_classes)
        )
        # self.categories_model_fc = nn.Linear(num_feat, self.num_classes)
        print('Done.')
        # Create the smaller networks, one for each category
        print('Loading the smaller networks for the species...')
        self.mini_net_model = {}
        for cat in self.categories:
            self.mini_net_model[cat] = nn.Linear(num_feat, self.num_species[cat])
        print('Done.')

    def set_parameters(self, datasets, loaders, train_params, loss_function, cuda_avail=False):
        self.categories = ['Amphibia', 'Animalia', 'Mammalia', 'Reptilia']
        self.num_species = {'Actinopterygii': 53, 'Amphibia': 115, 'Animalia': 77, 'Arachnida': 56,
                            'Aves': 964, 'Chromista': 9, 'Fungi': 121, 'Insecta': 1021, 'Mammalia': 186,
                            'Mollusca': 93, 'Plantae': 2101, 'Protozoa': 4, 'Reptilia': 289}
        self.num_classes = len(self.categories)
        self.datasets = datasets
        self.loaders = loaders
        self.optimizer = train_params['optimizer']
        self.weight_decay = train_params['weight_decay']
        self.learning_rate = train_params['learning_rate']
        self.gamma = train_params['gamma']
        self.step_size = train_params['step_size']
        if loss_function == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function == 'l1':
            self.loss_function = nn.L1Loss()
        elif loss_function == 'nll':
            self.loss_function = nn.NLLLoss()
        elif loss_function == 'l2':
            self.loss_function = nn.MSELoss()
        else:
            raise AttributeError('Invalid choice of loss function')
        self.cuda = cuda_avail

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
        if self.cuda:
            self.feat_model.cuda()
            self.categories_model_fc.cuda()
            for cat in self.categories:
                self.mini_net_model[cat].cuda()
        print('Done.')

        if self.optimizer == 'sgd':
            optimizer = optim.SGD(model.fc.parameters(), lr=self.learning_rate, momentum=0.9,
                                  weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            optimizer = optim.Adam(model.fc.parameters(), lr=self.learning_rate,
                                   weight_decay=self.weight_decay)
        elif self.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(model.fc.parameters(), lr=self.learning_rate, momentum=0.9,
                                      weight_decay=self.weight_decay)
        else:
            raise AttributeError('Invalid choice of optimizer')
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        # best_models = {what: copy.deepcopy(model.state_dict())}
        best_models = {what: model}
        best_acc = 0.0

        hist_acc = {'train': [], 'val': []}
        hist_loss = {'train': [], 'val': []}

        print('Starting training...')
        print('Size of the dataset is', len(self.datasets['train']))
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            batch_cnt = 0

            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)
                else:
                    model.train(False)

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, (supercategory_targets, species_targets) in self.loaders[phase]:
                    batch_cnt += 1
                    # wrap them in Variable
                    if self.cuda:
                        inputs = Variable(inputs.cuda())
                        if what == 'categories_net':
                            labels = Variable(supercategory_targets.cuda())
                        else:
                            labels = Variable(species_targets.cuda())
                    else:
                        inputs = Variable(inputs)
                        if what == 'categories_net':
                            labels = Variable(supercategory_targets)
                        else:
                            labels = Variable(species_targets)

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
                    if what is not 'categories_net' and phase == 'val':
                        running_corrects += self.__top_five__(outputs, labels.data)
                    else:
                        running_corrects += torch.sum(preds == labels.data)
                    batch_cnt += len(inputs)
                    progress = batch_cnt/len(self.datasets[phase]) * 100
                    print('%.2f' % progress, '%, Running loss is', running_loss, end='\r')

                epoch_loss = running_loss / len(self.datasets[phase])
                epoch_acc = running_corrects / len(self.datasets[phase])

                hist_acc[phase].append(epoch_acc)
                hist_loss[phase].append(epoch_loss)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # if best model, save it
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_models[what] = model

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model = best_models[what]
        return model, hist_acc, hist_loss

    def test(self):
        # Build the network
        print('Building the model...')
        if self.cuda:
            self.feat_model.cuda()
            self.categories_model_fc.cuda()
            for cat in self.categories:
                self.mini_net_model[cat].cuda()
        model = self.feat_model
        model.fc = self.categories_model_fc
        print('Done.')

        model.eval()
        correct_core = 0  # Number of correct predictions for the supercategories
        correct_species = 0  # Number of correct predictions for the single species
        print('Starting testing...')
        for data, (supercategories_targets, species_targets) in self.loaders['test']:
            if self.cuda:
                data, supercategories_targets, species_targets = data.cuda(), supercategories_targets.cuda(), \
                                                                 species_targets.cuda()
            data, supercategories_targets, species_targets = Variable(data, volatile=True), \
                                                             Variable(supercategories_targets), \
                                                             Variable(species_targets)
            model.fc = self.categories_model_fc
            supercategory_outputs = np.argmax(torch.nn.functional.softmax(model(data), dim=0).data, axis=1)
            for index, output in enumerate(supercategory_outputs):
                # only if supercategory classification is correct check single species
                if output == supercategories_targets[index].int:
                    correct_core += 1
                    model.fc = self.mini_net_model[self.categories[output]]
                    species_output = model(data)
                    pred = species_output.data.max(1, keepdim=True)[1]
                    correct_species += pred.eq(species_targets.data.view_as(pred)).cpu().sum()

        print('\nTest set: Accuracy on core network: {}/{} ({:.0f}%)\n'.format(correct_core,
                                                                               len(self.loaders['test'].dataset),
                                                                               100. * correct_core /
                                                                               len(self.loaders['test'].dataset)))
        print('\nTest set: Accuracy on branch networks: {}/{} ({:.0f}%)\n'.format(correct_species,
                                                                                  len(self.loaders['test'].dataset),
                                                                                  100. * correct_species /
                                                                                  len(self.loaders['test'].dataset)))

    def load_model(self, model, what):
        if what == 'categories_net':
            self.categories_model_fc = model
        else:
            self.mini_net_model[what] = model

    def __top_five__(self, outputs, labels):
        corrects = 0
        probabilities = torch.nn.functional.softmax(outputs, dim=0).data
        for index, el in enumerate(probabilities):
            for cnt in range(5):
                current_pred = np.argmax(el)
                if current_pred == labels[index]:
                    corrects += 1
                else:
                    probabilities[current_pred] = 0
        return corrects
