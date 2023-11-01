import shutil
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
import argparse
from matplotlib.patches import Rectangle, Polygon

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

def rotate_image_with_alpha(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # RGB 채널과 알파 채널을 개별적으로 회전
    rotated_rgb = cv2.warpAffine(img[:, :, :3], M, (w, h))
    
    # 알파 채널을 2D로 변환
    alpha_channel_2d = img[:, :, 3]
    rotated_alpha_2d = cv2.warpAffine(alpha_channel_2d, M, (w, h))
    rotated_alpha = rotated_alpha_2d[:, :, np.newaxis]  # 다시 3D로 변환
    
    # RGB 채널과 알파 채널을 다시 합침
    rotated_img = np.concatenate((rotated_rgb, rotated_alpha), axis=2)
    
    return rotated_img



def rotate_image_and_label(img_path, label_path, output_img_path, output_label_path, angle, position):
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
     # 알파 채널 검사 (4번째 채널)
    if img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
    else:
        raise ValueError("Object image does not have an alpha channel")
    h, w = img.shape[:2]
    center = (w // 2, h // 2)    
    rotated_img = rotate_image_with_alpha(img, angle)

    base_filename = os.path.basename(img_path).split('.')[0]
    augmented_img_filename = os.path.join(output_dataset_path, 'images','train', base_filename + '_ro.png')
    filename = augmented_img_filename    
    split_filename = filename.rsplit("_ro", 1)[0]
    extension = filename.split(".")[-1]
    n = 1
    new_filename = f"{split_filename}_ro{n}.{extension}"
    while os.path.exists(new_filename):  # For demonstration, using a condition that always evaluates to True
        n += 1
        new_filename = f"{split_filename}_ro{n}.{extension}"
        
    cv2.imwrite(new_filename, rotated_img)

    new_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            corners = rotate_bbox(cx * w, cy * h, bw * w, bh * h, angle, w, h)     

            if(angle<0):
                new_bw = corners[1][0] - corners[3][0]
                new_bh = corners[2][1] - corners[0][1]
                new_cx = corners[1][0] - new_bw/2 
                new_cy = corners[2][1] - new_bh/2
                new_cx /=w
                new_cy /=h
                new_bw /=w
                new_bh /=h
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
                new_labels.append(f"{cls} {new_cx} {new_cy} {new_bw} {new_bh}")
            
   
    split_filename_label = os.path.basename(new_filename).split('.')[0]
    augmented_label_filename = os.path.join(output_dataset_path, 'labels', 'train', split_filename_label + '.txt')  
     
    with open(augmented_label_filename, 'w') as f:
        for label in new_labels:
            f.write(label + "\n")

# Adjust brightness
def adjust_brightness(img_path, label_path, output_img_path, output_label_path, angle, position):
    # 알파 채널을 포함하여 이미지를 로드
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    factor = 1
    if factor is None:
        factor = random.uniform(0.5, 1.2)
    
    # RGB 채널만 밝기 조정
    adjusted_rgb = cv2.convertScaleAbs(img[:, :, :3], alpha=factor, beta=0)
    
    # 알파 채널이 존재하는 경우 그대로 유지
    if img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        adjusted_img = np.dstack([adjusted_rgb, alpha_channel])
    else:
        adjusted_img = adjusted_rgb

    base_filename = os.path.basename(img_path).split('.')[0]
    augmented_img_filename = os.path.join(output_dataset_path, 'images', 'train', base_filename + '_br.png')
    filename = augmented_img_filename    
    split_filename = filename.rsplit("_br", 1)[0]
    extension = filename.split(".")[-1]
    n = 1
    new_filename = f"{split_filename}_br{n}.{extension}"
    while os.path.exists(new_filename):
        n += 1
        new_filename = f"{split_filename}_br{n}.{extension}"           
    cv2.imwrite(new_filename, adjusted_img)

    split_filename_label = os.path.basename(new_filename).split('.')[0]
    augmented_label_filename = os.path.join(output_dataset_path, 'labels', 'train', split_filename_label + '.txt')
    shutil.copy(label_path, augmented_label_filename)

# Apply cropout
def apply_cropout(img_path, label_path, output_img_path, output_label_path, angle, position):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    labels = []
    box_width_percent = 40
    base_filename = os.path.basename(img_path).split('.')[0]
    augmented_img_filename = os.path.join(output_dataset_path, 'images','train', base_filename + '_cp.png')
    filename = augmented_img_filename    
    split_filename = filename.rsplit("_cp", 1)[0]
    extension = filename.split(".")[-1]
    n = 1
    new_filename = f"{split_filename}_cp{n}.{extension}"
    while os.path.exists(new_filename):  # For demonstration, using a condition that always evaluates to True
        n += 1
        new_filename = f"{split_filename}_cp{n}.{extension}"

    # Load labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            labels.append([float(part) for part in parts])
        base_filename_label = os.path.basename(new_filename).split('.')[0]
    augmented_label_filename = os.path.join(output_dataset_path, 'labels', 'train', base_filename_label + '.txt')
    shutil.copy(label_path, augmented_label_filename)

    # Directly embed the cropout logic here
    for label in labels:
        cls, cx, cy, bw, bh = label
        x_center, y_center, box_w, box_h = int(cx*w), int(cy*h), int(bw*w), int(bh*h)
        x1, y1 = int(x_center - box_w/2), int(y_center - box_h/2)
        x2, y2 = int(x_center + box_w/2), int(y_center + box_h/2)

        if position == 'left':
            cutout_x1 = x1
            cutout_x2 = x1 + int(0.01 * box_width_percent * box_w)
            cutout_y1 = y1
            cutout_y2 = y2
        elif position == 'right':
            cutout_x1 = x2 - int(0.01 * box_width_percent * box_w)
            cutout_x2 = x2
            cutout_y1 = y1
            cutout_y2 = y2
        elif position == 'top':
            cutout_x1 = x1
            cutout_x2 = x2
            cutout_y1 = y1
            cutout_y2 = y1 + int(0.01 * box_width_percent * box_h)
        elif position == 'bottom':
            cutout_x1 = x1
            cutout_x2 = x2
            cutout_y1 = y2 - int(0.01 * box_width_percent * box_h)
            cutout_y2 = y2
        elif position == 'center':
            cutout_w = int(0.01 * box_width_percent * box_w)
            cutout_h = int(0.01 * box_width_percent * box_h)
            cutout_x1 = x_center - cutout_w // 2
            cutout_x2 = x_center + cutout_w // 2
            cutout_y1 = y_center - cutout_h // 2
            cutout_y2 = y_center + cutout_h // 2
        elif position =='random':
            positions = ['left', 'right', 'top', 'bottom']
            random_position = random.choice(positions)
            if random_position == 'left':
                cutout_x1 = x1
                cutout_x2 = x1 + int(0.01 * box_width_percent * box_w)
                cutout_y1 = y1
                cutout_y2 = y2
            elif random_position == 'right':
                cutout_x1 = x2 - int(0.01 * box_width_percent * box_w)
                cutout_x2 = x2
                cutout_y1 = y1
                cutout_y2 = y2
            elif random_position == 'top':
                cutout_x1 = x1
                cutout_x2 = x2
                cutout_y1 = y1
                cutout_y2 = y1 + int(0.01 * box_width_percent * box_h)
            elif random_position == 'bottom':
                cutout_x1 = x1
                cutout_x2 = x2
                cutout_y1 = y2 - int(0.01 * box_width_percent * box_h)
                cutout_y2 = y2
            elif random_position == 'center':
                cutout_w = int(0.01 * box_width_percent * box_w)
                cutout_h = int(0.01 * box_width_percent * box_h)
                cutout_x1 = x_center - cutout_w // 2
                cutout_x2 = x_center + cutout_w // 2
                cutout_y1 = y_center - cutout_h // 2
                cutout_y2 = y_center + cutout_h // 2

        else:
            cutout_w = random.randint(int(0.15 * box_w), int(0.4 * box_w))
            cutout_h = random.randint(int(0.15 * box_h), int(0.4 * box_h))
            cutout_x1 = random.randint(x1, x2 - cutout_w)
            cutout_y1 = random.randint(y1, y2 - cutout_h)
            cutout_x2 = cutout_x1 + cutout_w
            cutout_y2 = cutout_y1 + cutout_h
        # 알파 채널이 있는 경우, 알파 값을 0으로 설정하여 크롭아웃 부분을 투명하게 만듦
        if img.shape[2] == 4:
            img[cutout_y1:cutout_y2, cutout_x1:cutout_x2, :3] = 0
            img[cutout_y1:cutout_y2, cutout_x1:cutout_x2, 3] = 0
        else:
            img[cutout_y1:cutout_y2, cutout_x1:cutout_x2] = 0
        

    cv2.imwrite(new_filename, img)

def apply_augmentation_on_dataset(augmentation_functions, dataset_base_path, output_base_path,angle,position):
    img_folder = os.path.join(dataset_base_path, 'images', 'train')
    label_folder = os.path.join(dataset_base_path, 'labels', 'train')
    output_img_folder = os.path.join(output_base_path, 'images', 'train')
    output_label_folder = os.path.join(output_base_path, 'labels', 'train')

    for img_file in os.listdir(img_folder):
        if img_file.endswith('.png'):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_folder, img_file)
            label_path = os.path.join(label_folder, base_name + '.txt')
            output_img_path = os.path.join(output_img_folder, img_file)
            output_label_path = os.path.join(output_label_folder, base_name + '.txt')
            for func in augmentation_functions:
                func(img_path, label_path, output_img_path, output_label_path, angle,position)

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

def display_image_with_original_and_rotated_bboxes(img_path, original_label_path, rotated_label_path, title, angle, position):
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
            #Original Bbox(blue box)
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
    parser.add_argument('--data', type=str, default='./datasets', help='input Path to the dataset')
    parser.add_argument('--output', type=str, default='./datasets\output', help='output Path to the dataset')
    parser.add_argument('--display-original', action='store_true', help='Display the original image with bounding boxes')
    parser.add_argument('--position', type=str, default='', choices=['left', 'right', 'top', 'bottom', 'center','random'], help='Position of the cropout')
    parser.add_argument('--box_width_percent', type=int, default=30, help='Width of the cropout box as a percentage of image width (default: 30)')
    
    args = parser.parse_args()

    functions = []
    if args.rotate:
        functions.append(rotate_image_and_label)
    if args.brightness:
        functions.append(adjust_brightness)
    if args.cropout:
        functions.append(apply_cropout)

    position = args.position

    original_dataset_path = args.data
    output_dataset_path = args.output
        # Ensure the necessary directories exist
    ensure_dir(os.path.join(output_dataset_path, 'images', 'train'))
    ensure_dir(os.path.join(output_dataset_path, 'labels', 'train'))
    
    #angle = random.uniform(0,-20)
    angle = -90
    angle = -angle

    apply_augmentation_on_dataset(functions, original_dataset_path, output_dataset_path, angle,position)

    # 폴더에서 무작위로 이미지 파일 하나를 선택
    example_img_folder = os.path.join(os.getcwd(), output_dataset_path, 'images', 'train')
    print(example_img_folder)
    all_images = [img for img in os.listdir(example_img_folder) if img.endswith('.png')]
    random_image_name = random.choice(all_images)

    # 선택한 이미지에 대한 라벨 파일 이름을 결정
    random_label_name = os.path.splitext(random_image_name)[0] + '.txt'

    # 무작위로 선택된 이미지와 라벨 파일의 경로 설정
    random_img_path = os.path.join(example_img_folder, random_image_name)
    random_label_path = os.path.join(output_dataset_path, 'labels', 'train', random_label_name)
    random_original_label_path = os.path.join(original_dataset_path, 'labels', 'train', random_label_name)
    display_image_with_bboxes(random_img_path,random_label_path,"title")
    # 이미지 표시
    #display_image_with_original_and_rotated_bboxes(
    #    random_img_path, 
    #    random_original_label_path,
    #    random_label_path, 
    #    " Rotated yolo-Bbox (Red) vs Rotated Bbox (green)",
    #    angle
    #)

