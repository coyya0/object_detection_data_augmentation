import os
from PIL import Image


def adjust_bbox(bbox, width_ratio, height_ratio):
    """Adjust the bounding box according to the resize ratios."""
    return [
        bbox[0] * width_ratio,
        bbox[1] * height_ratio,
        bbox[2] * width_ratio,
        bbox[3] * height_ratio,
    ]


def resize_images_and_labels_in_subfolder(
    subfolder_name, image_folder, label_folder, target_size, resize_folder
):
    # Define paths
    input_image_path = os.path.join(image_folder, subfolder_name)
    input_label_path = os.path.join(label_folder, subfolder_name)
    output_image_path = os.path.join(resize_folder, "images", subfolder_name)
    output_label_path = os.path.join(resize_folder, "labels", subfolder_name)

    # Create output directories if they don't exist
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    if not os.path.exists(output_label_path):
        os.makedirs(output_label_path)

    # List all jpg files in the input image folder
    image_files = [f for f in os.listdir(input_image_path) if f.endswith(".jpg")]

    for image_file in image_files:
        image_filepath = os.path.join(input_image_path, image_file)
        label_file = image_file.replace(".jpg", ".txt")
        label_filepath = os.path.join(input_label_path, label_file)

        # Check if the corresponding label file exists
        if not os.path.exists(label_filepath):
            print(f"Missing label for image: {image_file} in {subfolder_name}")
            continue

        # Resize the image
        image = Image.open(image_filepath)
        original_width, original_height = image.size
        image = image.resize(target_size)
        new_width, new_height = target_size

        # Compute the resize ratio
        width_ratio = new_width / original_width
        height_ratio = new_height / original_height

        # Adjust the bounding boxes in the label file
        with open(label_filepath, "r") as f:
            labels = f.readlines()






        adjusted_labels = []
        for label in labels:
            parts = label.strip().split()
            # Assuming the label format is: "class_id xmin ymin xmax ymax"
            bbox = [float(p) for p in parts[1:]]
            adjusted_bbox = adjust_bbox(bbox, width_ratio, height_ratio)
            adjusted_labels.append(f"{parts[0]} {' '.join(map(str, adjusted_bbox))}")

        # Save the resized image and adjusted labels to the output folders
        image.save(os.path.join(output_image_path, image_file))
        with open(os.path.join(output_label_path, label_file), "w") as f:
            f.write("\n".join(adjusted_labels))


def resize_images_and_labels(image_folder, label_folder, target_size, resize_folder):
    for subfolder_name in ["train", "val"]:
        resize_images_and_labels_in_subfolder(
            subfolder_name, image_folder, label_folder, target_size, resize_folder
        )


# TO DO : input path
img_path = os.path.join(os.getcwd(), "images")
label_path = os.path.join(os.getcwd(), "labels")
resize_path = os.path.join(os.getcwd(), "resize_folder")
# Example usage
resize_images_and_labels(img_path, label_path, (640, 480), resize_path)
