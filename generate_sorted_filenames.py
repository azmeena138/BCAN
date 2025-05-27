import os

# Since you're in E:\sami_bcan\data\Flicker8k_Dataset,
# the Images folder is in the same directory
image_folder = 'Images'
output_file = 'filenames_sorted.txt'

# Get all .jpg files and sort them
filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

# Write sorted filenames to text file
with open(output_file, 'w') as f:
    for name in filenames:
        f.write(name + '\n')

print("Saved sorted filenames to:", output_file)
