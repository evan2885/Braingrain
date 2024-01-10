import os
import mimetypes

data_directory = "croptailor/oat_images/Additional data with labels/"

# Change your destination folder here
destination_folder = "C:/Users/eveli/Documents/Universitetskurser/Tillämpad bioinformatik/all_images/"

# Remove this line if you already have a folder for the images
os.makedirs(destination_folder)

def process_file(file_path, destination_file_path, chunk_size=9000):
    print(f"Copying file: {file_path} to {destination_file_path}") # Not necessary, but useful for checking if the program is running
    try:
        with open(file_path, 'rb') as source_file:
            with open(destination_file_path, 'wb') as destination_file:
                chunk = source_file.read(chunk_size)
                while chunk:
                    destination_file.write(chunk)
                    chunk = source_file.read(chunk_size)
    except Exception as e:
        print(f"Error copying file: {e}")

def process_files_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        destination_file_path = destination_folder + file_name
        file_type, encoding = mimetypes.guess_type(file_path)

        if file_type == 'image/jpeg':
            process_file(file_path, destination_file_path)

def process_files_in_all_folders(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            process_files_in_folder(folder_path)

process_files_in_all_folders(data_directory)
