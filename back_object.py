import cv2
import numpy as np
import os
import random
import shutil

def merge_images_with_alpha(object_img_path, background_img_path, output_img_path):
    # 이미지 불러오기
    obj_img = cv2.imread(object_img_path, cv2.IMREAD_UNCHANGED)
    bg_img = cv2.imread(background_img_path, cv2.IMREAD_COLOR)
    
    # 배경 이미지에 알파 채널이 없으면 추가
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    # 이미지 크기가 맞지 않다면 배경 이미지 리사이즈
    if obj_img.shape[:2] != bg_img.shape[:2]:
        bg_img = cv2.resize(bg_img, (obj_img.shape[1], obj_img.shape[0]))

    # 알파 채널 검사 (4번째 채널)
    if obj_img.shape[2] == 4:
        alpha_channel = obj_img[:, :, 3]
    else:
        raise ValueError("Object image does not have an alpha channel")

    # 알파 채널을 이용하여 배경과 합성
    for c in range(0, 3):
        bg_img[:, :, c] = bg_img[:, :, c] * (1 - alpha_channel / 255.0) + obj_img[:, :, c] * (alpha_channel / 255.0)

    # 배경의 알파 채널도 업데이트
    bg_img[:, :, 3] = (alpha_channel / 255.0) * 255 + (1 - alpha_channel / 255.0) * bg_img[:, :, 3]

    # 결과 이미지 저장
    cv2.imwrite(output_img_path, bg_img)


#object_img_folder = 'D:/bk/datasets/alpha_test'
#background_img_folder = 'D:/bk/datasets/background/images'
#output_folder = os.path.join(os.getcwd(),"alpha_merge_test")  # 현재 디렉토리

object_img_folder = os.path.join(os.getcwd(),"split_data/merge")
background_img_folder = os.path.join(os.getcwd(),"background/images")
output_folder = os.path.join(os.getcwd(),"split_data/merge_output")
object_images = [f for f in os.listdir(object_img_folder) if f.endswith('.png')]
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
background_images = [f for f in os.listdir(background_img_folder) if f.endswith('.png')]

for i, object_img_name in enumerate(object_images, start=1):
    object_img_path = os.path.join(object_img_folder, object_img_name)
    
    # 랜덤한 배경 이미지 선택
    background_img_name = random.choice(background_images)
    background_img_path = os.path.join(background_img_folder, background_img_name)
    
    output_img_path = os.path.join(output_folder, f'merged_image_{i}.png')
    
    merge_images_with_alpha(object_img_path, background_img_path, output_img_path)
    
    # txt 파일 복사
    txt_path_src = os.path.join(object_img_folder, object_img_name.replace('.png', '.txt'))
    if os.path.exists(txt_path_src):
        txt_path_dst = os.path.join(output_folder, f'merged_image_{i}.txt')
        shutil.copy(txt_path_src, txt_path_dst)
