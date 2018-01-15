import torch.utils.data as data
import os
import os.path
import skimage.io as io
import preprocessor


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

    def __init__(self, root, annotations, transform):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annotations)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        # print('Loading image', index)
        coco = self.coco
        img_id = self.ids[index]
        img_ref = self.coco.loadImgs(img_id)
        try:
            # imgs = [io.imread(self.root + img_ref[i]['file_name']) for i, img in enumerate(img_ref)]
            img = io.imread(self.root + img_ref[0]['file_name'])
            # imgs = [preprocessor.normalize(img) for img in imgs]
            # print(type(imgs[0]))
            img = preprocessor.normalize(img)
            if self.transform:
                img = self.transform(img)

            ann_id = coco.getAnnIds(imgIds=img_id)
            ann = coco.loadAnns(ann_id)
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
        return len(self.ids)

    def get_size(self):
        return self.__len__()
