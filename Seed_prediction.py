import sys
sys.path.append('C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master')

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
from PIL import Image, ImageDraw

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

original_image = Image.open("C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/seeds_transfer_learning/DSC01352.JPG")
resized_image = original_image.resize((600, 400))
resized_image.save("C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/Test_pic_Seed.JPG")

CLASS_NAMES = ['BG', 'seed']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_NMS_THRESHOLD = 0.6

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/seeds_transfer_learning/transfer/transfer_seed_mask_rcnn_trained.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
#image = cv2.imread("/Users/torbjornfries/Desktop/Slutkurs/mrcnn4TF_2/Mask-RCNN-TF2/all_images/IMG_8554.JPG")
#image = cv2.imread("/Users/torbjornfries/Desktop/Slutkurs/croptailor/oat_images/Exempeldata/IMG_9291.JPG")
#image = cv2.imread("C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/Test_pic_Seed.JPG")
image = cv2.imread("C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/seeds_transfer_learning/IMG_9275.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores']
                                  )

#mask_image = Image.new("RGB", original_image.size, (0, 0, 0))
#draw = ImageDraw.Draw(mask_image)

#for i in range(r['masks'].shape[2]):
    #mask = r['masks'][:, :, i]
    #color = (255, 255, 255)  # White color
    #draw.bitmap((0, 0), Image.fromarray(mask.astype('uint8') * 255), fill=color)

#mask_areas = []

#for i in range(r['masks'].shape[2]):
   # mask = r['masks'][:, :, i]
    #area = sum(sum(mask))
   # mask_areas.append(area)

#print("Mask Areas:", mask_areas)

#print(len(mask_areas))
#mask_image.show()

