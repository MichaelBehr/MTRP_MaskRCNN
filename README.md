# MTRP_MaskRCNN

## 1. Overview
Using the implementation of Mask R-CNN on Python 3, Keras, and TensorFlow from: https://github.com/matterport/Mask_RCNN to detect/segment myofascial trigger points in ultrasound images.

## 2. Data
The 2D-CNN is scripted to be trained on a set of images with bounding boxes already extracted (where the Trigger Point is). Scripts currently look in the folder locations 'D:\Python Scripts\MTRP CNN\IMAGES' for the raw images, and 'D:\Python Scripts\MTRP CNN\MASKS' for the binary mask images. 

## 3. Running Scripts
Running the scripts is simple, and just requires you to set up your directory structure properly (to make sure the scripts access your training data). If this is done:

1. Run the training script and observe the results.
2. If they are to your liking run the test script.
3. Results from the test script demonstrate how well the 2D-CNN detected the trigger points in new images.

## 4. To-do
* Automate Mask R-CNN tuning (currently manual)
* Collect more data
* Run processes on more powerful GPU setup to improve model
