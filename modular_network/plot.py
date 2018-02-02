"""
This script plots the history of accuracies and losses provided as inputs
"""

import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='dl4cv_team50 Modular Network')
parser.add_argument('--filename', default='', metavar='F',
                    help='the name of the file with the info to be plotted')
args = parser.parse_args()

results = pickle.load(open(args.filename, 'rb'))
accuracy = results['accuracy']
train_loss = results['loss']['train']
val_loss = results['loss']['val']
train_acc = results['accuracy']['train']
val_acc = results['accuracy']['val']

e = np.arange(0, len(train_loss), 1)

# plt.plot(e, train_loss, 'r--', e, val_loss, 'b--')
# plt.ylabel('Loss')

plt.plot(e, train_acc, 'r', e, val_acc, 'b')
plt.ylabel('Accuracy')
plt.show()