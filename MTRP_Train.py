# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:14:32 2019

@author: behrM
"""

# fit a mask rcnn on our Trigger Point dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import skimage.io
import numpy as np

# class that defines and loads the MTRP dataset
class MTRPDataset(Dataset):
	# load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "MTRP")
        # define data locations
        images_dir = dataset_dir + '\IMAGES'
        masks_dir = dataset_dir + '\MASKS'
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            # skip all images after 58 if we are building the train set
            if is_train and int(image_id) >= 58:
                continue
            # skip all images before 58 if we are building the test/val set
            if not is_train and int(image_id) < 58:
                continue
            img_path = images_dir + '\\' + filename
            ann_path = masks_dir + '\\' + image_id
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
            
    # Redefine load_image function
    def load_image(self, image_id):
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
		# load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height
    # load the masks for an image
    def load_mask(self, image_id):
        # Load mask
        mask = []
        mask_dir = self.image_info[image_id]['annotation']+'.jpg'
        mask.append(skimage.io.imread(mask_dir,as_gray=True))
        mask = np.stack(mask, axis=-1)

        mask = (mask > 200).astype(bool)
#        # If has an alpha channel, remove it for consistency
#        if masks.shape[-1] == 4:
#            masks = masks[..., :3]
        class_ids = list()
        class_ids.append(self.class_names.index('MTRP'))
        return mask, asarray(class_ids, dtype='int32')
 
	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
 
# define a configuration for the model
class MTRPConfig(Config):
    # define the name of the configuration
    NAME = "MTRP_cfg"
    # number of classes (background + MTrP)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    VALIDATION_STEPS = 100
    TRAIN_ROIS_PER_IMAGE = 64
    STEPS_PER_EPOCH = 400
    BACKBONE = "resnet101"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    MAX_GT_INSTANCES = 16
    IMAGE_CHANNEL_COUNT = 3
    
 
# prepare train set
train_set = MTRPDataset()
train_set.load_dataset('D:\Python Scripts\MTRP CNN', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = MTRPDataset()
test_set.load_dataset('D:\Python Scripts\MTRP CNN', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = MTRPConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
