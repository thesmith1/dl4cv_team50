"""
Demo file for usage of the Preprocessor class
Initialize the object with the required parameters, then
call the process_images() method
"""

import sys, os
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from preprocessing.preprocessor import Preprocessor

data_dir = './data/'
annotations_dir = './annotations/'
dest_dir = './data_preprocessed/'
include_test_set = False

train_annotations = '{}train2017.json'.format(annotations_dir)
val_annotations = '{}val2017.json'.format(annotations_dir)
if include_test_set:
    test_annotations = '{}test2017.json'.format(annotations_dir)

preprocessor_train = Preprocessor(data_dir, train_annotations)
preprocessor_train.process_images(dest_dir)
preprocessor_val = Preprocessor(data_dir, val_annotations)
preprocessor_val.process_images(dest_dir)
if include_test_set:
    preprocessor_test = Preprocessor(data_dir, test_annotations)
    preprocessor_test.process_images(dest_dir)

print("Preprocessing in", dest_dir, "completed.")
