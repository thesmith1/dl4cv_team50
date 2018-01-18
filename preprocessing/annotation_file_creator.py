"""
script used to produce a reduced version (some species only) of the complete dataset on JSON
"""

import json

# Insert the subset of species to be selected below:
species_to_be_kept = ['Acinonyx jubatus', 'Lycaon pictus', 'Zalophus californianus']

# insert paths here
src_annotations_train = '../annotations/train2017.json'
src_annotations_val = '../annotations/val2017.json'

dst_annotations_train = '../annotations/train2017_min.json'
dst_annotations_val = '../annotations/val2017_min.json'


def new_annotation(old_annotation):
    old_target = old_annotation['category_id']
    new_target = subset_categories_indexes.index(old_target)
    result = dict(old_annotation)
    result['category_id'] = new_target
    return result


def new_category(old_category):
    old_id = old_category['id']
    new_id = subset_categories_indexes.index(old_id)
    result = dict(old_category)
    result['id'] = new_id
    return result

# load data sets
train_set = json.load(open(src_annotations_train))
val_set = json.load(open(src_annotations_val))
print('loaded files...', end='')

# obtain all annotations
train_annotations = train_set['annotations']
val_annotations = val_set['annotations']

# transform species names to indexes
subset_categories_indexes = [cat['id'] for cat in train_set['categories'] if cat['name'] in species_to_be_kept]

# obtain image ids for the images the required categories
filtered_image_ids_train = [ann['image_id'] for ann in train_annotations if ann['category_id'] in subset_categories_indexes]
filtered_image_ids_val = [ann['image_id'] for ann in val_annotations if ann['category_id'] in subset_categories_indexes]

# obtain images on the required categories (i.e. id in filtered image ids)
filtered_imgs_train = [img for img in train_set['images'] if img['id'] in filtered_image_ids_train]
filtered_imgs_val = [img for img in val_set['images'] if img['id'] in filtered_image_ids_val]

# keep only annotations of the required categories (i.e. id in filtered image ids),
# apply change of annotations to be in [0, K-1]
new_ann_train = [new_annotation(ann) for ann in train_set['annotations'] if ann['category_id'] in subset_categories_indexes]
new_ann_val = [new_annotation(ann) for ann in val_set['annotations'] if ann['category_id'] in subset_categories_indexes]

# assign new images
train_set['images'] = filtered_imgs_train
val_set['images'] = filtered_imgs_val
train_set['annotations'] = new_ann_train
val_set['annotations'] = new_ann_val

# assign new categories
present_cat_ids_train = {ann['category_id'] for ann in train_set['annotations']}
present_cat_ids_val = {ann['category_id'] for ann in val_set['annotations']}
new_cat_train = [new_category(cat) for cat in train_set['categories'] if cat['id'] in subset_categories_indexes]
new_cat_val = [new_category(cat) for cat in val_set['categories'] if cat['id'] in subset_categories_indexes]

train_set['categories'] = new_cat_train
val_set['categories'] = new_cat_val

# print(train_set['categories'])

# save
with open(dst_annotations_train, 'w') as train_file:
    json.dump(train_set, train_file)
with open(dst_annotations_val, 'w') as val_file:
    json.dump(val_set, val_file)

print("saved new files.")
