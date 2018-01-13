# dl4cv_team50

## General guidelines for the usage of the repo
The structure of the repo is the following (refer to this for the relative paths in your scripts and DON'T change it):
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
