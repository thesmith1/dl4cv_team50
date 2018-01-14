import torch.utils.data as data
import os
import os.path
import skimage.io as io
import preprocessor

class INaturalist(data.Dataset):
	'''
	The iNaturalist dataset object class
	Args:
		root (string): root directory where the images are (i.e. data/)
		annotations (string): path to the json annotations file
	Public methods:
		get_image: loads a single image; returns the image normalized and its label
		size: returns the size of the total dataset
	'''
	def __init__(self, root, annotations):
		from pycocotools.coco import COCO
		self.root = os.path.expanduser(root)
		self.coco = COCO(annotations)
		self.ids = list(self.coco.imgs.keys())

	def __getitem__(self, index):
		coco = self.coco
		img_id = self.ids[index]
		img_ref = self.coco.loadImgs(img_id)
		img = io.imread(self.root + img_ref[0]['file_name'])
		print('Here')
		img = preprocessor.normalize(img)

		ann_ids = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(ann_ids)
		target = [ann['category_id'] for ann in anns]

		return img, target

	def get_image(self, index):
		return self.__getitem__(index)

	def __len__(self):
		return len(self.ids)

	def size(self):
		return self.__len__()