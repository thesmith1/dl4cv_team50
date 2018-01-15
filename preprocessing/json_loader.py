'''
script used to produce a reduced version of the complete dataset on JSON
'''


import json

train_file = json.load(open('../annotations/train2017.json'))
val_file = json.load(open('../annotations/val2017.json'))
print('loaded files')
train_annotations = train_file['annotations']
val_annotations = val_file['annotations']

all_ids = [ann['image_id'] for ann in train_annotations if ann['category_id'] in [4985, 792, 4515]]
all_ids.extend([ann['image_id'] for ann in val_annotations if ann['category_id'] in [4985, 792, 4515]])

new_imgs_train = [img for img in train_file['images'] if img['id'] in all_ids]
new_imgs_val = [img for img in val_file['images'] if img['id'] in all_ids]

new_ann_train = [ann for ann in train_file['annotations'] if ann['image_id'] in all_ids]
new_ann_val = [ann for ann in val_file['annotations'] if ann['image_id'] in all_ids]

train_file['images'] = new_imgs_train
val_file['images'] = new_imgs_val
train_file['annotations'] = new_ann_train
val_file['annotations'] = new_ann_val

present_cat_ids_train = {ann['category_id'] for ann in train_file['annotations']}
present_cat_ids_val = {ann['category_id'] for ann in val_file['annotations']}
new_cat_train = [cat for cat in train_file['categories'] if cat['id'] in present_cat_ids_train]
new_cat_val = [cat for cat in val_file['categories'] if cat['id'] in present_cat_ids_val]

train_file['categories'] = new_cat_train
val_file['categories'] = new_cat_val

with open('../annotations/train2017_partial.json', 'w') as new_file:
	json.dump(train_file, new_file)

with open('../annotations/val2017_partial.json', 'w') as new_file:
	json.dump(val_file, new_file)

print("saved new files")