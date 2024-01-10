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

"""""""""""""""""""""""""""""
Instructions
För en testbild i samma size som trainingdata kör denna i nått script:

original_image = Image.open("PATH TILL /11 random pictures from same sample/IMG_9293.JPG")
resized_image = original_image.resize((600, 400))
resized_image.save("ÖNSKAD PATH/Test_pic_Seed.JPG")

Lägg in denna path i Seed_prediction
"""""""""""""""""""""""""""""

data_dir = "C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/FITTA"

class SeedDataset(utils.Dataset):
    def load_dataset(self, data_dir, is_train=True):
        self.add_class("dataset", 1, "seed")
        image_dirs = next(os.walk(data_dir))[1]

        for image_dir in image_dirs:
            image_id = image_dir
            """
            if is_train and not image_id.startswith('DSC'):
                continue
            if not is_train and image_id.startswith('DSC'):
                continue
            """
           
            img_path = os.path.join(data_dir, image_dir, 'images', f"{image_id}.jpg")
            masks_folder = os.path.join(data_dir, image_dir, 'masks')
                #print('img:', os.path.exists(img_path))
                #print('mask:', os.path.exists(masks_folder))
            mask_files = [file_name for file_name in os.listdir(masks_folder) if file_name.endswith('.png')]
            mask_paths = [os.path.join(masks_folder, mask_file) for mask_file in mask_files]
            self.add_image('dataset', image_id=image_id, path=img_path, mask_paths=mask_paths)
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_paths = info['mask_paths']
        
        masks = []
        for mask_path in mask_paths:
            mask = skimage.io.imread(mask_path).astype(bool)
            
            if mask.ndim > 2:
                mask = mask[..., 0]

            masks.append(mask)

        masks = np.stack(masks, axis=-1)
        class_ids = np.ones(masks.shape[-1], dtype=np.int32)

        return masks, class_ids
    
class SeedConfig(Config):
    NAME = "seed_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  
    STEPS_PER_EPOCH = 240  #öka till 130 eller nått
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

seed_dataset = SeedDataset()
seed_dataset.load_dataset(data_dir, is_train=True)
seed_dataset.prepare()

validation_dataset = SeedDataset()
validation_dataset.load_dataset(data_dir, is_train=False)
validation_dataset.prepare()

seed_config = SeedConfig()

model = MaskRCNN(mode='training', model_dir='./', config=seed_config)
model.load_weights(filepath='C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/seeds_transfer_learning/transfer/2x_mask_box_box.h5', by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=seed_dataset, val_dataset=validation_dataset, learning_rate=seed_config.LEARNING_RATE, epochs=2, layers='heads')


model_path = 'C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/seeds_transfer_learning/transfer/augmented_mask_box_box.h5'

model.keras_model.save_weights(model_path)
