#!/usr/bin/env python3

import json

import numpy as np
import pylab
from sklearn.manifold import TSNE
import skimage
from skimage.feature import hog
from skimage import data, exposure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys, os
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torchvision import models, transforms
from torch.autograd import Variable
import copy

## PARAMETERS FOR THE SCRIPT ##




#TSNE dims

no_dim = 2





#Label type for the datapoints

label_tpye = 0 # 1 for species, 0 for supercategory




#Parameter for loading a given NumPy array

load_npy = True




#Parameters for creating new annotations and image sets

make_new_annotations = False

annotations_dir = './annotations/'
name_of_annotation_file = 'vis_annotations.json'
train_annotations = '{}{}'.format(annotations_dir, name_of_annotation_file)


#species_to_be_kept = ['Acinonyx jubatus','Abaeis nicippe']
species_to_be_kept = [
'Acinonyx jubatus'
, 'Lycaon pictus'
, 'Zalophus californianus'
,'Acropora palmata'
]
supercategory_to_be_kept = None
supercategories_to_be_kept = None

SPECIFIC_SPECIES = 0
SPECIFIC_SUPERCATEGORY = 1
SET_OF_SUPERCATEGORIES = 2

mode = SPECIFIC_SPECIES

# destination paths

dst_annotations_train = './annotations/{}'.format(name_of_annotation_file)

# input path

src_annotations_train = './annotations/train2017.json'


def species_of_supecategory(dataset, current_supercategory):
    filt_species = [category for category in dataset['categories'] if
                    category['supercategory'] == current_supercategory]
    return [species['name'] for species in filt_species]



def get_tsne_figure(train_annotations, species_to_be_kept, supercategory_to_be_kept, supercategories_to_be_kept):
    if make_new_annotations == True:
        # load data sets
        train_set = json.load(open(src_annotations_train))
        print('loaded files...', end='')

        if mode == SPECIFIC_SUPERCATEGORY:
            species_to_be_kept = species_of_supecategory(train_set, supercategory_to_be_kept)
        elif mode == SET_OF_SUPERCATEGORIES:
            species_to_be_kept = []
            for supercategory in supercategories_to_be_kept:
                current_species = species_of_supecategory(train_set, supercategory)
                for species in current_species:
                    species_to_be_kept.append(species)

        # obtain all annotations
        train_annotations = train_set['annotations']

        # transform species to be kept to indexes
        subset_categories_indexes = [cat['id'] for cat in train_set['categories'] if cat['name'] in species_to_be_kept]
        print(subset_categories_indexes)

        # obtain image ids for the images the required categories
        filtered_image_ids_train = [ann['image_id'] for ann in train_annotations if
                                    ann['category_id'] in subset_categories_indexes]
    
        # obtain images on the required categories (i.e. id in filtered image ids)
        filtered_imgs_train = [img for img in train_set['images'] if img['id'] in filtered_image_ids_train]
    
        # keep only annotations of the required categories (i.e. id in filtered image ids)
        new_ann_train = [ann for ann in train_set['annotations'] if
                         ann['category_id'] in subset_categories_indexes]
    
        # assign new images
        train_set['images'] = filtered_imgs_train
        train_set['annotations'] = new_ann_train
    
        # assign new categories
        present_cat_ids_train = {ann['category_id'] for ann in train_set['annotations']}
        new_cat_train = [cat for cat in train_set['categories'] if cat['id'] in subset_categories_indexes]
    
        train_set['categories'] = new_cat_train
    
        print(train_set['categories'])
    
        # save
        with open(dst_annotations_train, 'w') as train_file:
            json.dump(train_set, train_file)
    
        print("saved new files.")




        print("Starting preprocessing of the images.")


        data_preprocess_dir = './data/'
        annotations_preprocess_dir = './annotations/'
        dest_preprocess_dir = './data_preprocessed_{}/'.format(pixel_per_axis)

        train_annotations = '{}{}'.format(annotations_preprocess_dir,name_of_annotation_file)

        preprocessor_train = Preprocessor(data_preprocess_dir, train_annotations, (pixel_per_axis, pixel_per_axis))
        preprocessor_train.process_images(dest_preprocess_dir)


        print("Preprocessing in", dest_preprocess_dir, "completed.")

        print("data dir is: ", data_dir)
        print("train annotations is: ", train_annotations)

    labels = None
    stacked = None
    if load_npy == True:
        labels = np.load("labels.npy")
        stacked = np.load("images.npy")

    else:

        print("\n\nLoading visualization set...")
        inaturalist_train = INaturalistDataset(data_dir, train_annotations, transform=None,
                                           modular_network_remap=False)


        stacked = np.zeros((len(inaturalist_train), number_of_features )) 
        labels = np.zeros(len(inaturalist_train))

        if number_of_channels == 1:
            for i, photo_data in enumerate(inaturalist_train):
    
                grey = skimage.color.rgb2grey(np.array(photo_data[0]))
    
                fd, hog_image = hog(grey, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                stacked[i] = np.array(hog_image_rescaled).flatten()
                labels[i] = photo_data[1][label_tpye]
        else:
            for i, photo_data in enumerate(inaturalist_train):
        
                stacked[i] = np.array(np.array(photo_data[0])).flatten()
                labels[i] = photo_data[1][label_tpye]


    #Perform TSNE

    X_embedded = TSNE(n_components=no_dim).fit_transform(stacked)



    #Plot results

    if no_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        x = X_embedded[:, 0]
        y = X_embedded[:, 1]
        z = X_embedded[:, 2]
        c = labels
    
        ax.scatter(x, y, z, c=c, cmap=plt.hot())
        plt.show()
    elif no_dim == 2:
        pylab.scatter(X_embedded[:, 0], X_embedded[:, 1], 20, labels)
        pylab.show()







if __name__ == '__main__':

    if len(sys.argv) == 3:   #Recieve number of pixels per axis and number of channels as command line arguments
        pixel_per_axis = int(sys.argv[1])
        number_of_channels = int(sys.argv[2])
        number_of_features = (pixel_per_axis**2) * number_of_channels
        data_dir = './data_preprocessed_{}/'.format(pixel_per_axis)
        get_tsne_figure(train_annotations, species_to_be_kept, supercategory_to_be_kept,supercategories_to_be_kept)
    else:    #Recieve given NumPy arrays for the data and the labels
        get_tsne_figure(train_annotations, species_to_be_kept, supercategory_to_be_kept, supercategories_to_be_kept)


