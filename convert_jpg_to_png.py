from PIL import Image
import os
from tqdm import tqdm

def jpg_to_png_with_transparency(jpg_file, png_file, transparency=128):
    img = Image.open(jpg_file).convert('RGBA')
    img_data = img.getdata()
    new_data = []
    for item in img_data:
        new_data.append(item[:3] + (transparency,))
    img.putdata(new_data)
    img.save(png_file, "PNG")

def convert_folder_jpg_to_png(input_folder, output_folder, transparency=128):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    jpg_files = [file_name for file_name in os.listdir(input_folder) if file_name.lower().endswith('.jpg')]
    for file_name in tqdm(jpg_files, desc='Converting JPG to PNG'): 
            jpg_file = os.path.join(input_folder, file_name)
            png_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.png')
            jpg_to_png_with_transparency(jpg_file, png_file, transparency)

# 사용 예시
input_folder = os.path.join(os.getcwd(),'images/val_jpg') # 변환할 폴더 경로
output_folder = os.path.join(os.getcwd(),'images/val')  # 저장할 폴더 경로
transparency = 255  # 투명도 설정 (0~255, 0이 완전 투명, 255가 완전 불투명)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

convert_folder_jpg_to_png(input_folder, output_folder, transparency)
