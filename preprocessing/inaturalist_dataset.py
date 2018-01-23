import torch.utils.data as data
import os
import os.path
import skimage.io as io
from pycocotools.coco import COCO
from preprocessing import preprocessor


class LabelRemapper(object):
    def __init__(self, all_labels):
        self.single_classes = sorted(list(set(all_labels)))
        self.class_count = len(self.single_classes)
        self.class_map = {single_class: self.single_classes.index(single_class) for single_class in self.single_classes}

    def __getitem__(self, item):
        return self.class_map[item]


class INaturalistDataset(data.Dataset):
    """
    The iNaturalist dataset object class
    Args:
        root (string): root directory where the images are (i.e. data/)
        annotations (string): path to the json annotations file
    Public methods:
        get_image: loads a single image; returns the image normalized (between -1 and 1) and its label
        get_images: loads multiple images and their labels (same behavior of get_image)
        get_size: returns the size of the total dataset
    """

    def __init__(self, root, annotations, transform, modular_network_remap=True):

        self.root = os.path.expanduser(root)
        self.coco = COCO(annotations)
        self.transform = transform
        self.all_ids = list(self.coco.imgs.keys())
        self.modular_network_remap = modular_network_remap

        # produce supercategory remapper
        all_categories = self.coco.cats
        all_supercategories = {cat['supercategory'] for cat in all_categories.values()}
        self.supercat_remapper = LabelRemapper(all_supercategories)

        if modular_network_remap:
            # produce single category remappers, stored in a dict
            self.category_remappers = dict()
            for supercategory in all_supercategories:
                intra_category_ids = {cat['id'] for cat in all_categories.values() if
                                      cat['supercategory'] == supercategory}
                single_category_remapper = LabelRemapper(intra_category_ids)
                self.category_remappers[supercategory] = single_category_remapper
        else:
            all_category_ids = {cat['id'] for cat in all_categories.values()}
            self.category_remapper = LabelRemapper(all_category_ids)

    def __getitem__(self, index):
        # print('Loading image', index)

        # find image id given index
        coco = self.coco
        img_id = self.all_ids[index]

        # find image given image id
        img_ref = coco.loadImgs(img_id)

        try:
            # imgs = [io.imread(self.root + img_ref[i]['file_name']) for i, img in enumerate(img_ref)]
            img = io.imread(self.root + img_ref[0]['file_name'])
            # imgs = [preprocessor.normalize(img) for img in imgs]

            img = preprocessor.normalize(img)
            if self.transform:
                img = self.transform(img)

            # find annotation of the image
            ann_id = coco.getAnnIds(imgIds=img_id)
            ann = coco.loadAnns(ann_id)

            category_id = ann[0]['category_id']
            supercategory = coco.cats[category_id]['supercategory']
            supercategory_target = self.supercat_remapper[supercategory]

            if self.modular_network_remap:
                category_target = self.category_remappers[supercategory][category_id]
            else:
                category_target = self.category_remapper[category_id]

        except FileNotFoundError as e:
            print(e)
            img = None
            supercategory_target = None
            category_target = None

        return img, (supercategory_target, category_target)

    def get_image(self, index):
        return self[index]

    def get_images(self, indexes):
        return self[indexes]

    def __len__(self):
        return len(self.all_ids)

    def get_size(self):
        return self.__len__()
