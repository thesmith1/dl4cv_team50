"""
Script used to divide the training set in training set and test set.
"""

import json
from pycocotools.coco import COCO

# input path
src_annotations_train = './annotations/train2017.json'

# output path
dst_annotations_train = './annotations/train2017_new.json'
dst_annotations_test = './annotations/test2017_new.json'

# ratio
TEST_SET_RATIO = 0.1

coco = COCO(src_annotations_train)
test_annotations = []
test_images = []
train_annotations = []
train_images = []
for i, category in enumerate(coco.cats):

    print("Completed %d/%d (%.2f%%)" % (i, len(coco.cats), i/len(coco.cats)), end='\r')
    # keep only annotations referred to this specific category

    filt_image_ids = sorted(coco.catToImgs[category])

    image_count = len(filt_image_ids)
    test_image_count = int(TEST_SET_RATIO*image_count)

    # divide: test
    test_filt_image_ids = filt_image_ids[0:test_image_count]
    test_filt_images = [coco.imgs[image_id] for image_id in test_filt_image_ids]
    test_filt_annotations = [coco.imgToAnns[image_id] for image_id in test_filt_image_ids]
    test_annotations.extend(test_filt_annotations)
    test_images.extend(test_filt_images)

    # divide: train
    train_filt_image_ids = filt_image_ids[test_image_count:]
    train_filt_images = [coco.imgs[image_id] for image_id in train_filt_image_ids]
    train_filt_annotations = [coco.imgToAnns[image_id] for image_id in train_filt_image_ids]
    train_annotations.extend(train_filt_annotations)
    train_images.extend(train_filt_images)

# create test json
test_set = json.load(open(src_annotations_train))
test_set['images'] = test_images
test_set['annotations'] = test_annotations

# replace in train set
train_set = json.load(open(src_annotations_train))
train_set['images'] = train_images
train_set['annotations'] = train_annotations

# save
with open(dst_annotations_train, 'w') as train_file:
    json.dump(train_set, train_file)
with open(dst_annotations_test, 'w') as test_file:
    json.dump(test_set, test_file)
