from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize
import skimage as sk
import os

'''
Preprocessor class, resizes the images to the desired dimensions
and stores them in the desired output folder 
'''
class Preprocessor(object):
	def __init__(self, source_root, annotations_path):
		self.coco = COCO(annotations_path)
		self.source_root = source_root
		self.final_size = (200,200)
		self.image_set = None

	def process_images(self, destination_root):
		if not os.path.exists(destination_root):
			os.makedirs(destination_root)
		catIds = self.coco.getCatIds(['Acinonyx jubatus'])
		for cat in catIds:
			imgIds = self.coco.getImgIds(catIds=cat)
			print(len(imgIds))
			images_ref = self.coco.loadImgs(imgIds)
			print(len(images_ref))
			images = [io.imread(self.source_root + image_ref['file_name']) for image_ref in images_ref]
			print(images_ref[0]['file_name'])
			species_dir, _ = os.path.split(os.path.join(destination_root,images_ref[0]['file_name']))
			print(species_dir)
			supercat_dir, _ = os.path.split(species_dir)
			print(supercat_dir)
			if not os.path.exists(os.path.join(destination_root, supercat_dir)):
				os.makedirs(supercat_dir)
			if not os.path.exists(species_dir):
				os.makedirs(species_dir)

			processed_images = [self.process_single_image(img) for img in images]			
			for img, img_ref in zip(processed_images, images_ref):
				io.imsave(destination_root + img_ref['file_name'], img)

	def process_single_image(self, image):
		return resize(image, self.final_size)

	@staticmethod
	def set_final_size(final_size):
		self.final_size = final_size

''' Global function for normalization of the image '''
def normalize(image):
	return image/127.5 - 1