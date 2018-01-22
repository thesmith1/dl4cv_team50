"""
Demo file for usage of the Preprocessor class
Initialize the object with the required parameters, then
call the process_images() method
"""

import sys, os
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from preprocessing.preprocessor import Preprocessor

data_dir = './data_min/'
annotations_dir = './annotations/modular_network/Animalia/'
dest_dir = './data_min2/'

train_annotations = '{}train2017_min.json'.format(annotations_dir)
val_annotations = '{}val2017_min.json'.format(annotations_dir)

preprocessor_train = Preprocessor(data_dir, train_annotations)
preprocessor_train.process_images(dest_dir)
preprocessor_val = Preprocessor(data_dir, val_annotations)
preprocessor_val.process_images(dest_dir, )

print("Preprocessing in", dest_dir, "completed.")
