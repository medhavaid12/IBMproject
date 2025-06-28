
import os
import shutil
import random
from pathlib import Path
import yaml

# SETTINGS
dataset_dir = Path("a")  # Change to your dataset folder name
output_dir = Path("face_mask_dataset")
split_ratio = 0.8  # 80% train, 20% val

# Create output folders
for split in ['train', 'val']:
    (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Gather all image files
images = list((dataset_dir / "images").glob("*.jpg")) + list((dataset_dir / "images").glob("*.png"))
random.shuffle(images)

# Split into train/val
split_index = int(len(images) * split_ratio)
train_imgs = images[:split_index]
val_imgs = images[split_index:]

def copy_data(img_list, split):
    for img_path in img_list:
        label_path = dataset_dir / "labels" / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(img_path, output_dir / "images" / split / img_path.name)
            shutil.copy(label_path, output_dir / "labels" / split / label_path.name)

copy_data(train_imgs, 'train')
copy_data(val_imgs, 'val')

# Write data.yaml
label_names = ['mask', 'no_mask']  # Edit if more labels used
data_yaml = {
    'train': str((output_dir / 'images/train').resolve()),
    'val': str((output_dir / 'images/val').resolve()),
    'nc': len(label_names),
    'names': label_names
}
with open(output_dir / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("✅ Dataset ready for YOLOv8 in:", output_dir.resolve())

import os
import shutil
import random
from pathlib import Path
import yaml

# SETTINGS
dataset_dir = Path("a")  # Change to your dataset folder name
output_dir = Path("face_mask_dataset")
split_ratio = 0.8  # 80% train, 20% val

# Create output folders
for split in ['train', 'val']:
    (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Gather all image files
images = list((dataset_dir / "images").glob("*.jpg")) + list((dataset_dir / "images").glob("*.png"))
random.shuffle(images)

# Split into train/val
split_index = int(len(images) * split_ratio)
train_imgs = images[:split_index]
val_imgs = images[split_index:]

def copy_data(img_list, split):
    for img_path in img_list:
        label_path = dataset_dir / "labels" / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(img_path, output_dir / "images" / split / img_path.name)
            shutil.copy(label_path, output_dir / "labels" / split / label_path.name)

copy_data(train_imgs, 'train')
copy_data(val_imgs, 'val')

# Write data.yaml
label_names = ['mask', 'no_mask']  # Edit if more labels used
data_yaml = {
    'train': str((output_dir / 'images/train').resolve()),
    'val': str((output_dir / 'images/val').resolve()),
    'nc': len(label_names),
    'names': label_names
}
with open(output_dir / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("✅ Dataset ready for YOLOv8 in:", output_dir.resolve())

