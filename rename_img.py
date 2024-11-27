###################################
# rename the image.py
###################################

import os

# this is where you store the image
image_folder_path = 'C:/Users/lucas/OneDrive/Pictures/rename/'

# file name in integer
i = 417

for index, filename in enumerate(os.listdir(image_folder_path)):
    old_file_path = os.path.join(image_folder_path, filename)

    if os.path.isfile(old_file_path):
        file_name, file_extension = os.path.splitext(filename)

        new_file_name = f"img{i}{file_extension}"

        new_file_path = os.path.join(image_folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_file_name}'")
        i+=1

print("Renaming complete.")
