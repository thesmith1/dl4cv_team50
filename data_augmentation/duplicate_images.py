"""
script used to create new, augmented images to a image set. Run from root directory
"""

import copy
import os
import json
import torchvision.transforms as transforms
from PIL import Image

# source_root is the data set you want to expand. destination_root is where the newly made images end up.
# dst_annotations is where the new annotation file for the entire dataset (original images + new images) ends up
source_root = './data_preprocessed_224/'
destination_root = './data_preprocessed_224/'
dst_annotations = './annotations/augmented_train2017.json'

# This is the original annotation file for the training images you want to expand
src_annotations_train = './annotations/reduced_dataset_train2017.json'

# thresholds are the image counts below which the number of images are multiplied by the corresponding mult_value
threshholds = [20, 40, 60, 80, 100, 200, 300]
mult_values = [40, 20, 10, 5, 4, 3, 2]

rot_degrees = 30
contrast_factor = 0.5
brightness_factor = 0.4

# augment the current set of images
def augment_image(train_images, img_list, mult_index, category_id, aug_counter):
    annotation_list = []
    image_annotation_list = []
    for img in img_list:
        file_name = [current_img['file_name'] for current_img in train_images if current_img['id'] == img]
        image = Image.open(source_root + file_name[0])

        data_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=rot_degrees),
            transforms.ColorJitter(brightness=brightness_factor, contrast=contrast_factor),
        ])

        for i in range(mult_values[mult_index]):
            augmented_image = data_aug(image)
            augmented_file_name = (file_name[0].split('.')[0] + str(aug_counter) + '.jpg')
            
            new_directory = augmented_file_name.rsplit('/', 1)[0]
            
            if not os.path.exists(destination_root + new_directory):
                os.makedirs(destination_root + new_directory)

            augmented_image.save(destination_root + augmented_file_name)

            new_annotation = {'id': aug_counter, 'image_id': aug_counter, 'category_id': category_id}
            new_image_annotation = {'id': aug_counter, 'file_name': augmented_file_name}

            annotation_list.append(new_annotation)
            image_annotation_list.append(new_image_annotation)

            aug_counter += 1

    return annotation_list, image_annotation_list, aug_counter


if __name__ == '__main__':

    aug_counter = 700000

    # load data sets
    train_set = json.load(open(src_annotations_train))

    # obtain all annotations
    train_annotations = train_set['annotations']
    train_images = train_set['images']
    train_species = train_set['categories']

    total_new_annotations_list = copy.copy(train_annotations)
    total_new_image_annotations_list = copy.copy(train_images)

    # for each species
    for species in train_species:
        # the list of images of the current species
        imgs_of_species = [ann['image_id'] for ann in train_annotations if ann['category_id'] == species['id']]
        
        # decide what amount of augmentation to do
        did_augmentation = True
        new_ann = None
        new_image_ann = None
        if len(imgs_of_species) < threshholds[0]:
            new_ann, new_image_ann, aug_counter = augment_image(train_images, imgs_of_species, 0, species['id'], aug_counter)
        elif len(imgs_of_species) < threshholds[1]:
            new_ann, new_image_ann, aug_counter = augment_image(train_images, imgs_of_species, 1, species['id'], aug_counter)
        elif len(imgs_of_species) < threshholds[2]:
            new_ann, new_image_ann, aug_counter = augment_image(train_images, imgs_of_species, 2, species['id'], aug_counter)
        elif len(imgs_of_species) < threshholds[3]:
            new_ann, new_image_ann, aug_counter = augment_image(train_images, imgs_of_species, 3, species['id'], aug_counter)
        elif len(imgs_of_species) < threshholds[4]:
            new_ann, new_image_ann, aug_counter = augment_image(train_images, imgs_of_species, 4, species['id'], aug_counter)
        elif len(imgs_of_species) < threshholds[5]:
            new_ann, new_image_ann, aug_counter = augment_image(train_images, imgs_of_species, 5, species['id'], aug_counter)
        elif len(imgs_of_species) < threshholds[6]:
            new_ann, new_image_ann, aug_counter = augment_image(train_images, imgs_of_species, 6, species['id'], aug_counter)
        else:
            did_augmentation = False

        if did_augmentation:
            # append the new objects to the total list
            total_new_annotations_list.extend(new_ann)
            total_new_image_annotations_list.extend(new_image_ann)

    # replace in the original file the new objects
    train_set['annotations'] = total_new_annotations_list
    train_set['images'] = total_new_image_annotations_list

    # save the file as json
    with open(dst_annotations, 'w') as train_file:
        json.dump(train_set, train_file)
