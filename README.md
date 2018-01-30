# dl4cv_team50

## General guidelines for the usage of the repo
The structure of the repo is the following (refer to this for the relative paths in your scripts and please DON'T change it):
* annotations (this folder contains the JSON files for the annotated dataset, i.e. the labels)
* data (the dataset, the structure of the folders is consistent with the annotations) (of course, this folder is empty...)
* preprocessing (this folder contains code for preprocessing)
* docs (this folder is for documents of any kind)
* single_network (this folder contains the code of the single big network)
* modular_network (this folder contains the code of the modularized network)

## Requirements
* pycocotools
* pytorch
* requirements.txt of dl4cv

## Installation of requirements
### Pycocotools
Download and extract from https://github.com/cocodataset/cocoapi, enter directory PythonAPI, open a terminal and run
```sh
python setup.py build_ext install
```
### Other requirements

Assumed your virtual environment has all the packages listed in the file requirements.txt installed, you may still need some additional installations to make the preprocessing work:
```sh
sudo apt-get install python3-tk
sudo apt-get install scikit-image
```

## Logs
### 20/01/2018: Giorgio
A new script named modular_network.py contains the code for a ResNet50 which can be overfitted on the small dataset.
**Important**: all the scripts following our relative import paths convention must be run from the root directory of the repo.
The script now accepts arguments in input (only hyperparameters and settings), using argparser.

Next step: adapt that script to accept as arguments choices among optimizers and loss functions and implement the modularity with multiple networks.

### 21/01/2018: Giorgio
The class ModularNetwork now supports different loss functions and optimizers and implements the methods train() and test(): we are ready for training.

Next step: code for saving results of training and test.

### 25/01/2018: Giorgio
The core of the modular network has been trained, without data augmentation, here the results:
* train accuracy: 78%
* validation accuracy: 79%
* epochs: 1
* learning rate: 1e-3
* total time: 5h 30m

### 25/01/2018: Giorgio
Only 4 supercategories over 13 are chosen from now on, in order to speed up the training process. They are Reptilia, Mammalia, Animalia and Amphibia.
They have been chosen mainly because they together contribute to a reduction of the dataset to 1/7 of its size and because the ratio between their number of images and their number of species is high.
The generate new annotations files have been included in the annotations folder (they are called reduced_dataset_train2017.json)

### 30/01/2018: Giorgio
Up to now the max validation accuracy for the core network is 84.3% (model 9), while for the branches I have
* Amphibia: 44.1% (model 2)
* Animalia: 80.83% (model 1)
* Mammalia: 57.7% (model 1)
* Reptilia: 41.3% (model 2)

The training with two fc layers for the branches gave unsatisfactory results (in average 6%); next step: more data augmentation