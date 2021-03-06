from pycocotools.coco import COCO
from PIL import Image
import os

'''
Preprocessor class, resizes the images to the desired dimensions
and stores them in the desired output folder 
'''


class Preprocessor(object):
    def __init__(self, source_root, annotations_path, final_size=(224, 224)):
        self.coco = COCO(annotations_path)
        self.source_root = source_root
        self.final_size = final_size
        self.image_set = None

    def process_images(self, destination_root):

        print("Starting preprocessing in %s" % destination_root)
        # create final root folder
        if not os.path.exists(destination_root):
            os.makedirs(destination_root)

        catIds = self.coco.getCatIds()
        for i, cat in enumerate(catIds):

            print("Completed folders: %d/%d (%.2f%%) -- current category id: %d"
                  % (i, len(catIds), i/len(catIds)*100, cat), end='\r')
            img_ids = self.coco.getImgIds(catIds=cat)
            images_ref = self.coco.loadImgs(img_ids)
            species_dir, _ = os.path.split(os.path.join(destination_root, images_ref[0]['file_name']))
            supercat_dir, _ = os.path.split(species_dir)

            # create super category directory, if necessary
            if not os.path.exists(supercat_dir):
                os.makedirs(supercat_dir)

            # create category directory, if necessary
            if not os.path.exists(species_dir):
                os.makedirs(species_dir)

            for image_ref in images_ref:
                image = Image.open(self.source_root + image_ref['file_name'])
                processed_image = self.process_single_image(image)
                processed_image.save(destination_root + image_ref['file_name'])
                image.close()
        print("\ndone.")

    def process_single_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
            # print("Image converted!")
        return image.resize(self.final_size)

    def set_final_size(self, final_size):
        self.final_size = final_size


''' Global function for normalization of the image '''


def normalize(image):
    return image / 127.5 - 1
