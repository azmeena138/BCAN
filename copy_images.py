import os
import shutil

# Paths
source_dir = 'E:\\sami_bcan\\data\\Flicker8k_Dataset\\Images'
dest_train = 'E:\\sami_bcan\\data\\flickr8k_raw\\train\\images'
dest_dev = 'E:\\sami_bcan\\data\\flickr8k_raw\\dev\\images'
filename_path = 'E:\\sami_bcan\\data\\Flicker8k_Dataset\\filenames_sorted.txt'

# Read sorted filenames
with open(filename_path, 'r') as f:
    filenames = [line.strip() for line in f]

# Copy first 6000 to train
for fname in filenames[:6000]:
    shutil.copy(os.path.join(source_dir, fname), dest_train)

# Copy next 1000 to dev
for fname in filenames[6000:7000]:
    shutil.copy(os.path.join(source_dir, fname), dest_dev)

print("Images copied: 6000 → train, 1000 → dev")
