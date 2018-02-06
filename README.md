# dl4cv_team50

## General guidelines for the usage of the repo
The structure of the repo is the following (refer to this for the relative paths in your scripts and please DON'T change it):
* annotations (this folder contains the JSON files for the annotated dataset, i.e. the labels)
* data (the dataset, the structure of the folders is consistent with the annotations) (of course, this folder is empty...)
* preprocessing (this folder contains code for preprocessing and dataset manipulation in pytorch)
* docs (this folder is for documents of any kind)
* single_network (this folder contains the code of the single big network)
* modular_network (this folder contains the code of the modularized network)
* attention (this folder contains the code of the attention network)
* data_augmentation (this folder contains the script used to apply data augmentation to the data folder)
* visualization (this folder contain code for generating images of data visualization and some results)

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
```
## How to use
* Download and extract the dataset from https://github.com/visipedia/inat_comp#data and put it in a folder called 'data' as child of the root folder
* Preprocess the data using demo_preprocess.py
* Run any script using the preprocessed images and the relative annotation files
