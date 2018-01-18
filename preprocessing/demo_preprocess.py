"""
Demo file for usage of the Preprocessor class
Initialize the object with the required parameters, then
call the process_images() method
"""

from preprocessor import Preprocessor

data_dir = '../data/'
annotations_dir = '../annotations/'
dest_dir = '../data2/'

train_annotations = '{}train2017_min.json'.format(annotations_dir)
val_annotations = '{}val2017_min.json'.format(annotations_dir)

preprocessor_train = Preprocessor(data_dir, train_annotations)
preprocessor_train.process_images(dest_dir)
preprocessor_val = Preprocessor(data_dir, val_annotations)
preprocessor_val.process_images(dest_dir, )

print("Preprocessing in", dest_dir, "completed.")