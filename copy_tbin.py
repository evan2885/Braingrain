import os
import shutil


destination_path = "C:/Users/alvin/Desktop/Object Detection/Mask-RCNN-TF2-master/all_tbin"
source_path = "C:/Users/alvin/Desktop/croptailor/oat_images/Additional data with labels"

for root, dirs, files in os.walk(source_path):
    for file in files:
        # Check if the file ends with .tbin
        if file.endswith('.tbin'):
            # Build the full path of the source file
            source_file_path = os.path.join(root, file)

            # Build the full path of the destination file
            destination_file_path = os.path.join(destination_path, file)

            # Copy the file to the destination directory
            shutil.copy(source_file_path, destination_file_path)

print("Copying completed.")

