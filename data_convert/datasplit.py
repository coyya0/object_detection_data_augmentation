import os
import shutil
from sklearn.model_selection import train_test_split

# 이미지와 레이블의 디렉토리
img_dir = "1.BBox_manual_labeling/Images/001/"
label_dir = "1.BBox_manual_labeling/Labels_formal/001/"

# 파일 리스트 가져오기
img_files = os.listdir(img_dir)
label_files = os.listdir(label_dir)

# 파일 리스트를 train, test, val로 분할
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

img_train, img_test, label_train, label_test = train_test_split(
    img_files, label_files, test_size=1 - train_ratio
)
img_val, img_test, label_val, label_test = train_test_split(
    img_test, label_test, test_size=test_ratio / (test_ratio + val_ratio)
)


# 파일을 각 디렉토리로 이동
def move_files(files, src_dir, dst_dir):
    for file in files:
        shutil.move(os.path.join(src_dir, file), os.path.join(dst_dir, file))


move_files(img_train, img_dir, "datasets/images/train/")
move_files(label_train, label_dir, "datasets/labels/train/")
move_files(img_val, img_dir, "datasets/images/val/")
move_files(label_val, label_dir, "datasets/labels/val/")
move_files(img_test, img_dir, "datasets/images/test/")
move_files(label_test, label_dir, "datasets/labels/test/")
