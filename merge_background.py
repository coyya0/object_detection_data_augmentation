import shutil
import random
import cv2
import numpy as np
import os
def merge_images_without_border(object_img_path, label_path, background_img_path, output_img_path, output_label_path):
    # Load images
    obj_img = cv2.imread(object_img_path)
    bg_img = cv2.imread(background_img_path)

    # Convert the object image to grayscale
    gray_obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)

    # Find contours of the object (external contour)
    contours, _ = cv2.findContours(gray_obj_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an alpha mask for the object
    alpha_mask = np.zeros_like(gray_obj_img)
    cv2.drawContours(alpha_mask, contours, -1, (255), thickness=cv2.FILLED)

    # Invert the alpha mask
    alpha_mask_inv = cv2.bitwise_not(alpha_mask)

    # Extract the inner object region
    inner_obj = cv2.bitwise_and(obj_img, obj_img, mask=alpha_mask)

    # Resize background to match object size
    bg_img = cv2.resize(bg_img, (obj_img.shape[1], obj_img.shape[0]))

    # Extract the background region
    bg_region = cv2.bitwise_and(bg_img, bg_img, mask=alpha_mask_inv)

    # Combine the inner object and background region
    merged_img = cv2.add(inner_obj, bg_region)

    # Save the output image
    cv2.imwrite(output_img_path, merged_img)

    # No adjustment needed for labels as the image size did not change
    shutil.copy(label_path, output_label_path)

# Example usage
object_img_folder = os.path.join(os.getcwd(), "merge_og_can/")
background_img_folder = os.path.join(os.getcwd(), "background/")
output_folder = os.path.join(os.getcwd(), "obbg_test/")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load object and background image paths
object_img_paths = [os.path.join(object_img_folder, f) for f in os.listdir(object_img_folder) if f.endswith('.jpg')]
background_img_paths = [os.path.join(background_img_folder, f) for f in os.listdir(background_img_folder) if f.endswith('.jpg')]

# Merge randomly selected object images with random backgrounds
for idx, object_img_path in enumerate(object_img_paths):
    background_img_path = random.choice(background_img_paths)
    output_img_path = os.path.join(output_folder, f'merged_{idx}.jpg')
    label_path = object_img_path.replace('.jpg', '.txt')
    output_label_path = os.path.join(output_folder, f'merged_{idx}.txt')
    merge_images_without_border(object_img_path, label_path, background_img_path, output_img_path, output_label_path)
