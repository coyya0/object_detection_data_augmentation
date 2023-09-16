import os
from pathlib import Path
from PIL import Image

base_root = os.getcwd()
image_train_folder = os.path.join(base_root, "resize_folder/images/train/*.jpg")
image_val_folder = os.path.join(base_root, "resize_folder/images/val/*.jpg")
label_train_folder = os.path.join(base_root, "resize_folder/labels/train/")
label_val_folder = os.path.join(base_root, "resize_folder/labels/val/")

output_image_train_folder = os.path.join(base_root, "cropped/images/train/")
output_image_val_folder = os.path.join(base_root, "cropped/images/val/")
output_label_train_folder = os.path.join(base_root, "cropped/labels/train/")
output_label_val_folder = os.path.join(base_root, "cropped/labels/val/")


def crop_and_save_images_and_labels_modified(
    image_path, label_path, output_image_folder, output_label_folder
):
    # Create directories if they don't exist
    Path(output_image_folder).mkdir(parents=True, exist_ok=True)
    Path(output_label_folder).mkdir(parents=True, exist_ok=True)

    image_file = os.path.basename(image_path)
    label_file = os.path.basename(label_path)

    # Load image and label
    img = Image.open(image_path)

    with open(label_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            label = parts[0]
            x_center, y_center, width, height = map(float, parts[1:])

            # Convert YOLO format to pixel values
            x_center *= img.width
            y_center *= img.height
            width *= img.width
            height *= img.height

            # Get top-left and bottom-right coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            print(
                x_center - width / 2,
                y_center - height / 2,
                x_center + width / 2,
                y_center + height / 2,
            )
            print(x1, y1, x2, y2)
            # Crop image and save
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img_path = os.path.join(
                output_image_folder, image_file.replace(".jpg", f"_{idx}.jpg")
            )
            cropped_img.convert("RGB").save(
                cropped_img_path
            )  # Convert to RGB before saving

            # Save label
            cropped_label_path = os.path.join(
                output_label_folder, label_file.replace(".txt", f"_{idx}.txt")
            )
            with open(cropped_label_path, "w") as out_f:
                # Since the cropped image only contains the object, the bounding box will cover the entire image
                out_line = f"{label} 0.5 0.5 1 1\n"
                out_f.write(out_line)

    print("Images and labels have been processed and saved.")


crop_and_save_images_and_labels_modified(
    image_train_folder,
    label_train_folder,
    output_image_train_folder,
    output_label_train_folder,
)

# Display the cropped images
cropped_images = [
    Image.open(os.path.join(output_image_train_folder, f))
    for f in os.listdir(output_image_train_folder)
    if f.endswith(".jpg")
]
cropped_images
