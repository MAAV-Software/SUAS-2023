import os
import random
import shutil

def randomly_partition_images(source_folder, destination_folder, percentage=20):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    all_files = os.listdir(source_folder)

    num_files_to_move = int(len(all_files) * (percentage / 100.0))

    files_to_move = random.sample(all_files, num_files_to_move)
    for file_name in files_to_move:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)

if __name__ == "__main__":
    source_folder = ".data/training_images"
    destination_folder = ".data/test_images"
    percentage_to_move = 20
    randomly_partition_images(source_folder, destination_folder, percentage_to_move)

    print(f"{percentage_to_move}% of images moved to {destination_folder}")
