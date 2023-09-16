import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
import argparse
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.patches import Polygon

def rotate_point(origin, point, angle):
    ox, oy = origin  # image_center
    px, py = point   # bbox point 

    # Adjust the angle for counter-clockwise rotation
    angle = -angle

    # Translate the point to rotate about the origin
    translated_px = px - ox
    translated_py = py - oy

    # Rotate the translated point
    rotated_px = translated_px * np.cos(np.radians(angle)) - translated_py * np.sin(np.radians(angle))
    rotated_py = translated_px * np.sin(np.radians(angle)) + translated_py * np.cos(np.radians(angle))
    # Translate the point back to its original location
    qx = rotated_px + ox
    qy = rotated_py + oy
    #print(f"ox = {ox} px = {px} t_px = {translated_px} r_px = {rotated_px} qx = {qx}")
    
    return qx, qy


def rotate_bbox(cx, cy, w, h, angle, img_w, img_h):
    center = (img_w / 2, img_h / 2)
    corners = [
        [cx - w / 2, cy - h / 2], # left top d 
        [cx + w / 2, cy - h / 2], # right top c
        [cx + w / 2, cy + h / 2], # right bottom b
        [cx - w / 2, cy + h / 2]  # left bottom a
    ]
    rotated_corners = [rotate_point(center, corner, angle) for corner in corners]
    return rotated_corners

def rotate_image_and_label(img_path, label_path, output_img_path, output_label_path, angle):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite(output_img_path, rotated_img)

    new_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            corners = rotate_bbox(cx * w, cy * h, bw * w, bh * h, angle, w, h)
            
          # Draw the rotated bounding box on the image
          #  for i in range(4):
          #      cv2.line(rotated_img, (int(corners[i][0]), int(corners[i][1])), 
          #                  (int(corners[(i+1)%4][0]), int(corners[(i+1)%4][1])), (0,255,0), 2)  # Green color for the bounding box

         # Display the image with the bounding box
           # img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
            #plt.imshow(img_rgb)
            #plt.axis('off')
            #plt.show()
            if(angle<0):
                #print("angle > 0 ")
                new_bw = corners[1][0] - corners[3][0]
                #print(f"{corners[1][0]} {corners[3][0]}")
                new_bh = corners[2][1] - corners[0][1]
                new_cx = corners[1][0] - new_bw/2 
                new_cy = corners[2][1] - new_bh/2
                new_cx /=w
                new_cy /=h
                new_bw /=w
                new_bh /=h
                #print(f"{cls} {new_cx} {new_cy} {new_bw} {new_bh}")
                new_labels.append(f"{cls} {new_cx} {new_cy} {new_bw} {new_bh}")

            else :       
                print("angle < 0\n")                                     
                new_bw = corners[2][0] - corners[0][0]
                new_bh = corners[3][1] - corners[1][1]
                new_cx = corners[2][0] - new_bw/2 
                new_cy = corners[3][1] - new_bh/2
                new_cx /=w
                new_cy /=h
                new_bw /=w
                new_bh /=h
                #print(f"{cls} {new_cx} {new_cy} {new_bw} {new_bh}")
                new_labels.append(f"{cls} {new_cx} {new_cy} {new_bw} {new_bh}")
            
            # if co

    with open(output_label_path, 'w') as f:
        for label in new_labels:
            f.write(label + "\n")

# Adjust brightness
def adjust_brightness(img_path, label_path, output_img_path,output_label_path, angle):
    img = cv2.imread(img_path)
    
    #factor = 0.7
    if factor is None:
        factor = random.uniform(0.5, 1.2)
    print(f" factor = {factor}")
    adjusted_img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    cv2.imwrite(output_img_path, adjusted_img)

# Apply cropout
def apply_cropout(img_path, label_path, output_img_path, output_label_path, angle):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    labels = []
    
    # Load labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            labels.append([float(part) for part in parts])

    # Directly embed the cropout logic here
    for label in labels:
        cls, cx, cy, bw, bh = label
        x_center, y_center, box_w, box_h = int(cx*w), int(cy*h), int(bw*w), int(bh*h)
        x1, y1 = int(x_center - box_w/2), int(y_center - box_h/2)
        x2, y2 = int(x_center + box_w/2), int(y_center + box_h/2)

        # Calculate the area to black out
        cutout_w = random.randint(int(0.1 * box_w), int(0.3 * box_w))
        cutout_h = random.randint(int(0.1 * box_h), int(0.3 * box_h))
        cutout_x1 = random.randint(x1, x2 - cutout_w)
        cutout_y1 = random.randint(y1, y2 - cutout_h)
        cutout_x2 = cutout_x1 + cutout_w
        cutout_y2 = cutout_y1 + cutout_h

        img[cutout_y1:cutout_y2, cutout_x1:cutout_x2] = 0

    cv2.imwrite(output_img_path, img)

def apply_augmentation_on_dataset(augmentation_functions, dataset_base_path, output_base_path,angle):
    img_folder = os.path.join(dataset_base_path, 'images', 'train')
    label_folder = os.path.join(dataset_base_path, 'labels', 'train')
    output_img_folder = os.path.join(output_base_path, 'images', 'train')
    output_label_folder = os.path.join(output_base_path, 'labels', 'train')

    for img_file in os.listdir(img_folder):
        if img_file.endswith('.jpg'):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_folder, img_file)
            label_path = os.path.join(label_folder, base_name + '.txt')
            output_img_path = os.path.join(output_img_folder, img_file)
            output_label_path = os.path.join(output_label_folder, base_name + '.txt')
            for func in augmentation_functions:
                func(img_path, label_path, output_img_path, output_label_path, angle)

def display_image_with_bboxes(img_path, label_path, title):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read the label file and draw the bounding boxes
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x = (cx - bw / 2) * img.shape[1]
            y = (cy - bh / 2) * img.shape[0]
            width = bw * img.shape[1]
            height = bh * img.shape[0]
            plt.gca().add_patch(Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none'))

            

    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def ensure_dir(directory):
    """
    Given path, create the directory (and any necessary parent directories) if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def display_image_with_original_and_rotated_bboxes(img_path, original_label_path, rotated_label_path, title, angle):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw original bounding boxes in blue
    with open(original_label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            corners = rotate_bbox(cx * w, cy * h, bw * w, bh * h, angle, w, h)
            x = (cx - bw / 2) * img.shape[1]
            y = (cy - bh / 2) * img.shape[0]
            width = bw * img.shape[1]
            height = bh * img.shape[0]
            #plt.gca().add_patch(Rectangle((x, y), width, height, linewidth=1, edgecolor='b', facecolor='none'))

            # Draw the rotated bounding box on the image (green box)
            polygon = Polygon(corners, closed=True, edgecolor='green', fill=None, linewidth=1)
            plt.gca().add_patch(polygon)

    # Draw rotated bounding boxes in red
    with open(rotated_label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x = (cx - bw / 2) * img.shape[1]
            y = (cy - bh / 2) * img.shape[0]
            width = bw * img.shape[1]
            height = bh * img.shape[0]
            plt.gca().add_patch(Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none'))

    

    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply augmentations to the dataset")
    parser.add_argument('--rotate', action='store_true', help='Apply rotation augmentation')
    parser.add_argument('--brightness', action='store_true', help='Apply brightness adjustment')
    parser.add_argument('--cropout', action='store_true', help='Apply cropout augmentation')
    parser.add_argument('--data', type=str, default='./', help='Path to the dataset')
    parser.add_argument('--display-original', action='store_true', help='Display the original image with bounding boxes')
    args = parser.parse_args()

    functions = []
    if args.rotate:
        functions.append(rotate_image_and_label)
    if args.brightness:
        functions.append(adjust_brightness)
    if args.cropout:
        functions.append(apply_cropout)

    original_dataset_path = args.data
    output_dataset_path = os.path.join(args.data, 'augmented_dataset')
        # Ensure the necessary directories exist
    ensure_dir(os.path.join(output_dataset_path, 'images', 'train'))
    ensure_dir(os.path.join(output_dataset_path, 'labels', 'train'))

    angle = random.uniform(30,45)
    #print(angle)
    apply_augmentation_on_dataset(functions, original_dataset_path, output_dataset_path, angle)

    # 폴더에서 무작위로 이미지 파일 하나를 선택
    example_img_folder = os.path.join(output_dataset_path, 'images', 'train')
    all_images = [img for img in os.listdir(example_img_folder) if img.endswith('.jpg')]
    random_image_name = random.choice(all_images)

    # 선택한 이미지에 대한 라벨 파일 이름을 결정
    random_label_name = os.path.splitext(random_image_name)[0] + '.txt'

    # 무작위로 선택된 이미지와 라벨 파일의 경로 설정
    random_img_path = os.path.join(example_img_folder, random_image_name)
    random_label_path = os.path.join(output_dataset_path, 'labels', 'train', random_label_name)
    random_original_label_path = os.path.join(original_dataset_path, 'labels', 'train', random_label_name)

    # 이미지 표시
    display_image_with_original_and_rotated_bboxes(
        random_img_path, 
        random_original_label_path,
        random_label_path, 
        " Rotated yolo-Bbox (Red) vs Rotated Bbox (green)",
        angle
    )

"""
    # Display example images
    example_img_folder = os.path.join(output_dataset_path, 'images', 'train')
    example_label_folder = os.path.join(output_dataset_path, 'labels', 'train')
    example_images = os.listdir(example_img_folder)


    
    
    # Show the first image as an example for each augmentation
    if example_images:
        example_img_path = os.path.join(example_img_folder, example_images[0])
        example_label_path = os.path.join(example_label_folder, os.path.splitext(example_images[0])[0] + '.txt')
        
        img = cv2.imread(example_img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read the label file and draw the bounding boxes
        with open(example_label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                cls, cx, cy, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x = (cx - bw / 2) * img.shape[1]
                y = (cy - bh / 2) * img.shape[0]
                width = bw * img.shape[1]
                height = bh * img.shape[0]
                plt.gca().add_patch(Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none'))

        plt.imshow(img_rgb)
        plt.title("Example Image with Applied Augmentations")
        plt.axis('off')
        plt.show()"""