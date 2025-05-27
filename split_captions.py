import random
from collections import defaultdict

def split_captions_by_image(input_file="Flickr8k.token.txt"):
    """
    Reads captions from the input file, groups them by image,
    and splits them into train.txt (5000 images, 25k captions)
    and test.txt (1000 images, 5k captions), maintaining the 5 captions per image.
    """
    image_captions = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # The first line is the header, skip it for processing data but include in output files
    header = lines[0]
    data_lines = lines[1:]

    for line in data_lines:
        parts = line.strip().split(',', 1) # Split only on the first comma
        if len(parts) == 2:
            image_filename = parts[0].strip()
            caption_text = parts[1].strip()
            image_captions[image_filename].append(line) # Store the original line including newline

    # Get a list of all unique image filenames
    image_filenames = list(image_captions.keys())
    random.shuffle(image_filenames) # Shuffle the images themselves

    # Determine split counts
    # For Flickr8k, there are 8000 images in total.
    # 5000 images for train (5000 * 5 = 25000 captions)
    # 1000 images for dev (1000 * 5 = 5000 captions) - We'll use this for test
    # 1000 images for test (1000 * 5 = 5000 captions) - You might have separate test/dev splits in your dataset.
    # Based on your request for 25k in train and 5k in test:
    # This implies 5000 images for train and 1000 images for test.
    
    num_train_images = 5000
    num_test_images = 1000 # Remaining images for test

    # Select images for train and test sets
    train_image_files = image_filenames[:num_train_images]
    test_image_files = image_filenames[num_train_images:num_train_images + num_test_images]

    # Collect all captions for the selected train images
    train_output_lines = []
    for img_file in train_image_files:
        train_output_lines.extend(image_captions[img_file])

    # Collect all captions for the selected test images
    test_output_lines = []
    for img_file in test_image_files:
        test_output_lines.extend(image_captions[img_file])

    # Write to train.txt
    with open("train.txt", 'w', encoding='utf-8') as f_train:
        f_train.write(header)
        f_train.writelines(train_output_lines)

    # Write to test.txt
    with open("test.txt", 'w', encoding='utf-8') as f_test:
        f_test.write(header)
        f_test.writelines(test_output_lines)

    print(f"Successfully processed {len(image_filenames)} unique images.")
    print(f"Train set: {len(train_image_files)} images, {len(train_output_lines)} captions.")
    print(f"Test set: {len(test_image_files)} images, {len(test_output_lines)} captions.")

# Call the function to perform the split
split_captions_by_image()