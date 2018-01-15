'''
Demo file for usage of the Preprocessor class
Initialize the object with the required parameters, then
call the process_images() method
'''

from preprocessor import Preprocessor

data_dir = '../data/'
annotations_dir = '../annotations/'
train_annotations = '{}train2017.json'.format(annotations_dir)
val_annotations = '{}val2017.json'.format(annotations_dir)

preprocessor_train = Preprocessor(data_dir, train_annotations)
preprocessor_train.process_images('../data2/', ['Acinonyx jubatus', 'Lycaon pictus', 'Zalophus californianus'])
preprocessor_val = Preprocessor(data_dir, val_annotations)
preprocessor_val.process_images('../data2/', ['Acinonyx jubatus', 'Lycaon pictus', 'Zalophus californianus'])