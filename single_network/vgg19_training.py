"""
script for training a VGG-19 model, necessary for visualization.
"""

import sys
import os
from torchvision import models, transforms
import torch
import torch.optim as optim
from torch.autograd import Variable

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
# print(sys.path)
from preprocessing.inaturalist_dataset import INaturalistDataset

# paths
data_dir = './data_preprocessed_224/'
annotations_dir = './annotations/'
train_annotations = '{}augmented_train2017.json'.format(annotations_dir)
val_annotations = '{}reduced_dataset_val2017.json'.format(annotations_dir)
test_annotations = '{}reduced_dataset_test2017.json'.format(annotations_dir)

# hyper-parameters
learning_rate = 1e-3
regularization_strength = 0
batch_size = 128
num_epochs = 10
loss = torch.nn.CrossEntropyLoss

# other parameters
validation_during_training = True
do_testing = True
applied_transformations = transforms.Compose([transforms.ToTensor()])
non_printable = ["model", "optimizer", "loss"]
cuda = torch.cuda.is_available()
log_interval = 1


def setup_vgg19(parameters, output_categories=667):

    # get pre-trained model, change classifier layers
    model = models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # move model to GPU
    if cuda:
        model = model.cuda()

    # create optimizer
    adam = optim.Adam(model.classifier.parameters(), lr=parameters['lr'], weight_decay=parameters['reg'])
    return model, adam


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


if __name__ == '__main__':

    # init
    params = {'lr':learning_rate, 'reg':regularization_strength}

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
    parameters['model'] = models.vgg19
    parameters['lr'] = learning_rate
    parameters['reg'] = regularization_strength
    parameters['batch-size'] = batch_size
    parameters['num-epochs'] = num_epochs
    parameters['optimizer'] = optim.Adam
    parameters['loss'] = loss
    parameters['output-filename'] = "{0}.pth".format(
        "_".join([str(key) + "=" + (parameter.__name__ if key in non_printable else str(parameter))
                  for key, parameter in parameters.items()]))

    print("\n\nSetting up model " + parameters['output-filename'])
    model, adam = setup_vgg19(params)
    loss = loss()

    # training
    print("Starting training (%d epoch%s)" % (num_epochs, "s" if num_epochs != 1 else ""))
    for epoch_count in range(1, num_epochs + 1):
        one_epoch_train(model, adam, loss, train_loader, epoch_count)
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
