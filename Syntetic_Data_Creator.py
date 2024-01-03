import os
import cv2
from imgaug.augmenters import Sequential, Fliplr, MedianBlur, LinearContrast, BilateralBlur, Sometimes

data_path = '/Users/torbjornfries/Desktop/Slutkurs/mrcnn4TF_2/Mask-RCNN-TF2/Transfer-learning/Images_With_Masks'
output_folder = '/Users/torbjornfries/Desktop/Slutkurs/mrcnn4TF_2/Mask-RCNN-TF2/Transfer-learning/Augmented_Images_With_Masks'

augIms_per_image = 1 #Dont add too many, takes forever.. if only 1, then put Fliplr(1), otherwise put Fliplr(0.5) or something
folders2augment = 238 #Max = amount of images in "data_path"-folder, whish is obtained from Create_Several_masks2.py, 238 for me

def augment_data(folder_path, output_folder, augIms_per_image):
    subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
    
    augIms_created = 0
    augmented_imagefolders = 0

    for subfolder in subfolders:
        augIms_created = 0
        images_folder = os.path.join(folder_path, subfolder, 'images')
        masks_folder = os.path.join(folder_path, subfolder, 'masks')
        image_file = os.path.join(images_folder, subfolder + '.jpg')
        image = cv2.imread(image_file)

        masks = sorted([mask_file for mask_file in os.listdir(masks_folder) if mask_file.endswith('.png')])

        for i, mask_file in enumerate(masks):
            seq = Sequential([
                Fliplr(1), #flips image

                #Dont know about these, put low probability of occuring(0.1), maybe comment out if augIms_per_image = 1
                Sometimes(0.2, LinearContrast((1.2, 1.6))),

            ], random_order=True)
            augmented_img = seq(image=image)

            new_output_folder = os.path.join(output_folder, f"{subfolder}_Augmented_{i}")
            os.makedirs(new_output_folder, exist_ok=True)
            new_images_folder = os.path.join(new_output_folder, 'images')
            new_masks_folder = os.path.join(new_output_folder, 'masks')
            os.makedirs(new_images_folder, exist_ok=True)
            os.makedirs(new_masks_folder, exist_ok=True)

            output_image_path = os.path.join(new_images_folder, f"{subfolder}Augmented_{i}.jpg")
            cv2.imwrite(output_image_path, augmented_img)

            for j, mask_file in enumerate(masks):
                mask_path = os.path.join(masks_folder, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                augmented_mask = seq(image=mask)
                output_mask_path = os.path.join(new_masks_folder, f"{subfolder}Augmented_{i}_mask_{j}.png")
                cv2.imwrite(output_mask_path, augmented_mask)

            augIms_created += 1
            if augIms_created >= augIms_per_image:
                break
        augmented_imagefolders+=1
        if augmented_imagefolders >= folders2augment:
            break

augment_data(data_path, output_folder, augIms_per_image)