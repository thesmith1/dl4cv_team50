import torch.utils.data as data
import os
import os.path
import skimage.io as io
from preprocessing import preprocessor

supercategory_target = {'Actinopterygii': 0, 'Amphibia': 1, 'Animalia': 2, 'Arachnida': 3, 'Aves':4,
                    'Chromista': 5, 'Fungi': 6, 'Insecta': 7, 'Mammalia': 8, 'Mollusca': 9,
                    'Plantae': 10, 'Protozoa': 11, 'Reptilia': 12}


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

    def __init__(self, root, annotations, transform, classify_supercategories=False):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annotations)
        self.transform = transform
        self.all_ids = list(self.coco.imgs.keys())
        self.classify_supercategories = classify_supercategories

    def __getitem__(self, index):
        # print('Loading image', index)

        # find image id given index
        coco = self.coco
        img_id = self.all_ids[index]

        # find image given image id
        img_ref = self.coco.loadImgs(img_id)

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

            # depending on target mode, produce target (either supercategory or category)
            if self.classify_supercategories:
                category_id = ann[0]['category_id']
                supercategory = coco.cats[category_id]['supercategory']
                target = supercategory_target[supercategory]
            else:
                target = ann[0]['category_id']

        except FileNotFoundError as e:
            print(e)
            img = None
            target = None

        return img, target

    def get_image(self, index):
        return self[index]

    def get_images(self, indexes):
        return self[indexes]

    def __len__(self):
        return len(self.all_ids)

    def get_size(self):
        return self.__len__()
