import xml.etree
import os
from numpy import zeros, asarray
import sys
sys.path.append("C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master")
sys.path.append("C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/mrcnn")
import skimage.io
import mrcnn
from mrcnn import utils
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script directory
os.chdir(script_dir)


#old imports
#import mrcnn.utils
#import mrcnn.config
#import mrcnn.model

#changed from mrcnn.utils.Dataset
class SeedDataset(utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adds information (image ID, image path, and annotation file path) about each image in a dictionary.
        self.add_class("dataset", 1, "seeds")
        images_dir = dataset_dir + '/all_images/'
        annotations_dir = dataset_dir + '/annotations/'

        print("Current working directory:", os.getcwd())
        print("Images directory:", os.path.abspath(images_dir))

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            if is_train and not image_id.startswith('DSC'):
                continue

            if not is_train and image_id.startswith('DSC'):
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.JPG.xml'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('seeds'))
        return masks, asarray(class_ids, dtype='int32')

    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

#changed from mrcnn.config.Config
class SeedConfig(Config):
    NAME = "seed_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 2

    STEPS_PER_EPOCH = 243

# Train
train_dataset = SeedDataset()
train_dataset.load_dataset(dataset_dir='seeds', is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = SeedDataset()
validation_dataset.load_dataset(dataset_dir='seeds', is_train=False)
validation_dataset.prepare()

# Model Configuration
kangaroo_config = SeedConfig()

# Build the Mask R-CNN Model Architecture
#changed from mrcnn.model
model = MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=kangaroo_config)

model.log_dir = os.path.join(model.model_dir, "logs")

# Update model directory separator
model.model_dir = model.model_dir.replace('/', os.path.sep)

print("Log directory:", model.log_dir)
if not os.path.exists(model.log_dir):
    os.makedirs(model.log_dir, exist_ok=True)


model.load_weights(filepath='C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/seeds_transfer_learning/seed_mask_rcnn_trained.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=kangaroo_config.LEARNING_RATE, 
            epochs=5, 
            layers='heads')

model_path = 'transfer_seed_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)

#testar ändra 2279 från self.log_dir = "//logdir//train"
