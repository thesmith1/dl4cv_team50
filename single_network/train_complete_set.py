#!/usr/bin/env python3
"""
script with functions for training a monolithic model on the dataset
"""

import torch.nn as nn
import torch.utils.data
from torchvision import models
from torch.autograd import Variable
from math import sqrt

# script parameters
log_interval = 1
cuda = torch.cuda.is_available()


def setup_model(parameters, output_categories=667, num_fc_layers=1, train_last_conv_layers=False):
    # unwrap parameters
    chosen_model = parameters['model']
    chosen_optimizer = parameters['optimizer']
    lr = parameters['lr']

    # get pre-trained model
    model = chosen_model(pretrained=True)

    # set gradient requirement
    for param in model.parameters():
        param.requires_grad = False
    if train_last_conv_layers:
        if isinstance(model, models.Inception3):
            additional_trained_params = set.union(set(model.Mixed_7a.parameters()),
                               set(model.Mixed_7b.parameters()),
                               set(model.Mixed_7c.parameters()))
            for param in additional_trained_params:
                param.requires_grad = True
        elif isinstance(model, models.ResNet):
            additional_trained_params = set.union(set(model.layer3.parameters()), set(model.layer4.parameters()))
            for param in additional_trained_params:
                param.requires_grad = True
        else:
            additional_trained_params = {}
            raise ValueError("Invalid model type")
    else:
        additional_trained_params = {}

    # change FC part
    fc_in_features = model.fc.in_features
    # print("fc_in_features:", fc_in_features)
    if num_fc_layers == 1:
        model.fc = nn.Linear(fc_in_features, output_categories)
    elif num_fc_layers == 2:
        intermediate_fc_layer_size = int(sqrt(fc_in_features*output_categories))
        model.fc = nn.Sequential(nn.Linear(fc_in_features, intermediate_fc_layer_size),
                                 nn.ReLU(),
                                 nn.Linear(intermediate_fc_layer_size, output_categories))
    else:
        raise ValueError("Max number of layer tht can be added for now: 2, got %d" % num_fc_layers)

    # move model to GPU
    if cuda:
        model = model.cuda()

    # create optimizer
    trained_params = set.union(set(model.fc.parameters()), additional_trained_params)
    # print([(type(trained_param.data), trained_param.requires_grad) for trained_param in trained_params])
    optimizer = chosen_optimizer(trained_params, lr=lr, weight_decay=parameters['reg'])
    return model, optimizer


def correct_predictions(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).int().sum(0, keepdim=True)
        res.append(correct_k.data.cpu().numpy().squeeze().tolist())
    return res


# single epoch of training method
def one_epoch_train(model, optimizer, loss, train_loader, epoch):

    # set train mode
    model.train()

    # for each batch
    for batch_idx, (data, targets) in enumerate(train_loader):

        # keep only species target
        _, target = targets
        # print("Batch dim:", data.shape)
        data, (target) = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        # forward pass
        output = model(data)

        # compute loss(es)
        if isinstance(output, tuple):
            loss_value = sum(loss(output_i, target) for output_i in output)
        else:
            loss_value = loss(output, target)

        # backward pass
        loss_value.backward()

        # weight upgrade
        optimizer.step()

        # log
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * train_loader.batch_size,
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss_value.data[0]), end='\r')


def evaluate(model, loss, dataset_loader):
    # set evaluation mode
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
        output = model(data)

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


def save_model_statistics(output_filename, results_val, results_test=None):

    with open(output_filename, "w") as fp:
        print('Validation results\n'
              'final loss: %.2f, correct (top1): %d, correct (top5): %d, predicted: %d\n\n' % results_val, file=fp)

    if results_test is not None:
        print("Test results\n"
              "final loss: %.2f, correct (top1): %d, correct (top5): %d, predicted: %d\n\n" % results_test, file=fp)


def complete_train_validation(parameters, loaders, output_categories, validation_during_training=False):

    # unwrap parameters and set model up
    train_loader, val_loader, test_loader = loaders
    num_epochs = parameters['num-epochs']
    loss = parameters['loss']()
    model, optimizer = setup_model(parameters, output_categories, parameters['num-fc-layers'], parameters['train-conv'])

    # training
    print("Starting training (%d epoch%s)" % (num_epochs, "s" if num_epochs != 1 else ""))
    for epoch_count in range(1, num_epochs + 1):
        one_epoch_train(model, optimizer, loss, train_loader, epoch_count)
        if validation_during_training:
            evaluate(model, loss, val_loader)

    # evaluation on validation set
    print("\nEvaluating model on validation set...")
    results_val = evaluate(model, loss, val_loader)

    # evaluation on test set
    results_test = None
    if test_loader is not None:
        print("Evaluating model on test set...")
        results_test = evaluate(model, loss, test_loader)

    # saving
    model_output_file = "./single_network/models/" + parameters['output-filename']
    print("Saving model %s..." % model_output_file, end='')
    torch.save(model, model_output_file)
    save_model_statistics(model_output_file[:-4] + ".txt", results_val, results_test)
    print("done.")
