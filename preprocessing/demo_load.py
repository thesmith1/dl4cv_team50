'''
Demo script for loading the dataset in batches
'''

from inaturalist import INaturalist
import matplotlib.pyplot as plt

data_dir = '../data/'
annotations_dir = '../annotations/'
train_annotations = '{}train2017.json'.format(annotations_dir)
val_annotations = '{}val2017.json'.format(annotations_dir)

dataset = INaturalist(data_dir, train_annotations)
print(dataset.__len__())
index = 540719
image, target = dataset.get_image(index)
print(target)