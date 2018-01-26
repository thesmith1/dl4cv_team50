import argparse
import pickle
import matplotlib.pyplot as plt

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

plt.plot(train_loss)
plt.ylabel('Validation accuracy')
plt.show()
