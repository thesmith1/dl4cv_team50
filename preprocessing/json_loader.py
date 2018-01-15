"""
script used to produce a reduced version (3 categories only) of the complete dataset on JSON
"""

import json

subset_categories_indexes = [4985, 792, 4515]


def new_annotation(old_annotation):
    old_target = old_annotation['category_id']
    new_target = subset_categories_indexes.index(old_target)
    result = dict(old_annotation)
    result['category_id'] = new_target
    return result


# load data sets
train_set = json.load(open('../annotations/train2017.json'))
val_set = json.load(open('../annotations/val2017.json'))
print('loaded files...', end='')

# obtain labels of the required categories
train_annotations = train_set['annotations']
val_annotations = val_set['annotations']

# obtain image ids for the images the required categories
all_ids = [ann['image_id'] for ann in train_annotations if ann['category_id'] in subset_categories_indexes]
all_ids.extend([ann['image_id'] for ann in val_annotations if ann['category_id'] in subset_categories_indexes])

# keep only images on the required categories (i.e. id in all_ids)
new_imgs_train = [img for img in train_set['images'] if img['id'] in all_ids]
new_imgs_val = [img for img in val_set['images'] if img['id'] in all_ids]

# keep only annotations of the required categories (i.e. id in all_ids), apply change of annotations to be in [0, K-1]
new_ann_train = [new_annotation(ann) for ann in train_set['annotations'] if ann['image_id'] in all_ids]
new_ann_val = [new_annotation(ann) for ann in val_set['annotations'] if ann['image_id'] in all_ids]

train_set['images'] = new_imgs_train
val_set['images'] = new_imgs_val
train_set['annotations'] = new_ann_train
val_set['annotations'] = new_ann_val

present_cat_ids_train = {ann['category_id'] for ann in train_set['annotations']}
present_cat_ids_val = {ann['category_id'] for ann in val_set['annotations']}
new_cat_train = [cat for cat in train_set['categories'] if cat['id'] in present_cat_ids_train]
new_cat_val = [cat for cat in val_set['categories'] if cat['id'] in present_cat_ids_val]

train_set['categories'] = new_cat_train
val_set['categories'] = new_cat_val

# print(train_set['categories'])

# save
with open('../annotations/train2017_min.json', 'w') as train_file:
    json.dump(train_set, train_file)
with open('../annotations/val2017_min.json', 'w') as val_file:
    json.dump(val_set, val_file)

print("saved new files.")
