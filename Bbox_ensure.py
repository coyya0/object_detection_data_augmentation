import os
import cv2
from tqdm import tqdm

# Let's define the folder path where the uploaded image and label file are located
folder_path = os.path.join(os.getcwd(),'split_data/merge')
image_path = os.path.join(folder_path,'images/train')
label_path = os.path.join(folder_path, 'labels/train')

def adjust_bbox(x, y, w, h, img_w, img_h):
    # Calculate the coordinates of the top left and bottom right corners
    x_center = x * img_w
    y_center = y * img_h
    w_scaled = w * img_w
    h_scaled = h * img_h
    x1 = x_center - w_scaled / 2
    y1 = y_center - h_scaled / 2
    x2 = x_center + w_scaled / 2
    y2 = y_center + h_scaled / 2

    # Clamp the coordinates to the image boundaries
    clamped_x1 = max(x1, 0)
    clamped_y1 = max(y1, 0)
    clamped_x2 = min(x2, img_w)
    clamped_y2 = min(y2, img_h)

    # Calculate the area inside the image
    inside_area = max(0, clamped_x2 - clamped_x1) * max(0, clamped_y2 - clamped_y1)
    total_area = w_scaled * h_scaled

    # If more than 60% of the box is outside the image, return None
    if inside_area / total_area < 0.4:
        return None

    # Recalculate width and height
    w_new = clamped_x2 - clamped_x1
    h_new = clamped_y2 - clamped_y1

    # Recalculate center coordinates
    x_new = clamped_x1 + w_new / 2
    y_new = clamped_y1 + h_new / 2

    # Normalize the coordinates by the dimensions of the image
    return x_new / img_w, y_new / img_h, w_new / img_w, h_new / img_h

# Now, we will apply the bounding box adjustment to all .png and .txt file pairs in the folder_path


# ...
# ...
for filename in tqdm(os.listdir(folder_path), desc='Processing files', unit='file'):
    if filename.endswith('.png'):
        img_path = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0]
        current_label_path = os.path.join(folder_path, base_name + '.txt')

        # Load image to get dimensions
        image = cv2.imread(img_path)
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        # Check if corresponding label file exists
        if os.path.isfile(current_label_path):
            # Read label file
            with open(current_label_path, 'r') as file:
                lines = file.readlines()

            # Adjust bounding boxes
            adjusted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w, h = map(float, parts)
                    adjusted_bbox = adjust_bbox(x, y, w, h, img_w, img_h)
                    # Write the line if the bounding box is valid and less than 60% outside the image
                    if adjusted_bbox is not None:
                        x, y, w, h = adjusted_bbox
                        adjusted_lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            # Write the adjusted bounding boxes back to the label file only if there are adjusted lines
            if adjusted_lines:
                with open(current_label_path, 'w') as file:
                    file.writelines(adjusted_lines)


