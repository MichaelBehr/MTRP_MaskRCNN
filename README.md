# MTRP_MaskRCNN

## Summary
Using the implementation of Mask R-CNN on Python 3, Keras, and TensorFlow from: https://github.com/matterport/Mask_RCNN to detect/segment myofascial trigger points in ultrasound images.

## Data
The 2D-CNN is scripted to be trained on a set of images with bounding boxes already extracted (where the Trigger Point is). Scripts currently look in the folder locations 'D:\Python Scripts\MTRP CNN\IMAGES' for the raw images, and 'D:\Python Scripts\MTRP CNN\MASKS' for the binary mask images.
