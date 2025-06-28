
import os
import shutil
import random
from pathlib import Path

# === Paths ===
base_dir = Path("face_mask_dataset")  # change if your dataset is in another location
train_img_dir = base_dir / "images/train"
train_lbl_dir = base_dir / "labels/train"
val_img_dir = base_dir / "images/val"
val_lbl_dir = base_dir / "labels/val"

# === Create val folders if they don't exist ===
val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# === List all image files in train ===
image_files = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
random.shuffle(image_files)

# === Calculate 20% of total for validation ===
val_count = int(0.2 * len(image_files))
val_images = image_files[:val_count]

# === Move images and their matching labels to val ===
moved = 0
for img_path in val_images:
    label_path = train_lbl_dir / (img_path.stem + ".txt")

    if label_path.exists():
        # Move image and label to val/
        shutil.move(str(img_path), str(val_img_dir / img_path.name))
        shutil.move(str(label_path), str(val_lbl_dir / label_path.name))
        moved += 1
    else:
        print(f"⚠️ Skipping {img_path.name} - missing label file.")

print(f"✅ Moved {moved} image-label pairs to validation set.")

import os
import shutil
import random
from pathlib import Path

# === Paths ===
base_dir = Path("face_mask_dataset")  # change if your dataset is in another location
train_img_dir = base_dir / "images/train"
train_lbl_dir = base_dir / "labels/train"
val_img_dir = base_dir / "images/val"
val_lbl_dir = base_dir / "labels/val"

# === Create val folders if they don't exist ===
val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# === List all image files in train ===
image_files = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
random.shuffle(image_files)

# === Calculate 20% of total for validation ===
val_count = int(0.2 * len(image_files))
val_images = image_files[:val_count]

# === Move images and their matching labels to val ===
moved = 0
for img_path in val_images:
    label_path = train_lbl_dir / (img_path.stem + ".txt")

    if label_path.exists():
        # Move image and label to val/
        shutil.move(str(img_path), str(val_img_dir / img_path.name))
        shutil.move(str(label_path), str(val_lbl_dir / label_path.name))
        moved += 1
    else:
        print(f"⚠️ Skipping {img_path.name} - missing label file.")

print(f"✅ Moved {moved} image-label pairs to validation set.")

