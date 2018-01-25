"""
script used to produce a reduced version (some species only) of the complete dataset on JSON, for debugging purposes
"""

import json

SPECIFIC_SPECIES = 0
SPECIFIC_SUPERCATEGORY = 1

species_to_be_kept = []
supercategory_to_be_kept = None
dst_annotations_train = ''
dst_annotations_val = ''

'''
inputs
'''

# input paths
src_annotations_train = './annotations/train2017_new.json'
src_annotations_val = './annotations/val2017.json'
mode = SPECIFIC_SUPERCATEGORY

# option for specific modes
if mode == SPECIFIC_SPECIES:
    # Insert the subset of species to be selected below:
    species_to_be_kept = ['Actinemys marmorata', 'Hypsiglena jani', 'Zootoca vivipara']

    supercategory = 'Reptilia'
    dst_annotations_train = './annotations/modular_network/{}/train2017_min.json'.format(supercategory)
    dst_annotations_val = './annotations/modular_network/{}/val2017_min.json'.format(supercategory)

elif mode == SPECIFIC_SUPERCATEGORY:

    supercategory_to_be_kept = 'Chromista'

    dst_annotations_train = './annotations/modular_network/{}/train2017_min.json'.format(supercategory_to_be_kept)
    dst_annotations_val = './annotations/modular_network/{}/val2017_min.json'.format(supercategory_to_be_kept)
else:
    exit(-1)


def species_of_supecategory(dataset, supecategory):
    filt_species = [category for category in dataset['categories'] if category['supercategory'] == supecategory]
    return [species['name'] for species in filt_species]


if __name__ == '__main__':

    # load data sets
    train_set = json.load(open(src_annotations_train))
    val_set = json.load(open(src_annotations_val))
    print('loaded files...', end='')

    if mode == SPECIFIC_SUPERCATEGORY:
        species_to_be_kept = species_of_supecategory(train_set, supercategory_to_be_kept)

    # obtain all annotations
    train_annotations = train_set['annotations']
    val_annotations = val_set['annotations']

    # transform species to be kept to indexes
    subset_categories_indexes = [cat['id'] for cat in train_set['categories'] if cat['name'] in species_to_be_kept]

    # obtain image ids for the images the required categories
    filtered_image_ids_train = [ann['image_id'] for ann in train_annotations if
                                ann['category_id'] in subset_categories_indexes]
    filtered_image_ids_val = [ann['image_id'] for ann in val_annotations if ann['category_id'] in subset_categories_indexes]

    # obtain images on the required categories (i.e. id in filtered image ids)
    filtered_imgs_train = [img for img in train_set['images'] if img['id'] in filtered_image_ids_train]
    filtered_imgs_val = [img for img in val_set['images'] if img['id'] in filtered_image_ids_val]

    # keep only annotations of the required categories (i.e. id in filtered image ids)
    new_ann_train = [ann for ann in train_set['annotations'] if
                     ann['category_id'] in subset_categories_indexes]
    new_ann_val = [ann for ann in val_set['annotations'] if ann['category_id'] in subset_categories_indexes]

    # assign new images
    train_set['images'] = filtered_imgs_train
    val_set['images'] = filtered_imgs_val
    train_set['annotations'] = new_ann_train
    val_set['annotations'] = new_ann_val

    # assign new categories
    present_cat_ids_train = {ann['category_id'] for ann in train_set['annotations']}
    present_cat_ids_val = {ann['category_id'] for ann in val_set['annotations']}
    new_cat_train = [cat for cat in train_set['categories'] if cat['id'] in subset_categories_indexes]
    new_cat_val = [cat for cat in val_set['categories'] if cat['id'] in subset_categories_indexes]

    train_set['categories'] = new_cat_train
    val_set['categories'] = new_cat_val

    print(train_set['categories'])

    # save
    with open(dst_annotations_train, 'w') as train_file:
        json.dump(train_set, train_file)
    with open(dst_annotations_val, 'w') as val_file:
        json.dump(val_set, val_file)

    print("saved new files.")
