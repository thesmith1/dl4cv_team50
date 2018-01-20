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
## Logs
### 20/01/2018: Giorgio
A new script named modular_network.py contains the code for a ResNet50 which can be overfitted on the small dataset. Unfortunately, due to a severe problem with imports on python using relative paths, the only solution I found so far of making it work is to place it in the root directory, so now I'm using the modular_network folder as storage of the best models. Next step: adapt that script to accept arguments and implement the modularity with multiple networks.
