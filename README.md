# augmentation_final_updated.py
augmentation_final_updated.py을 사용하면 아래 3가지의 데이터 증강을 실행할 수 있습니다! 
아래의 parser를 확인해주세요 

    parser.add_argument('--rotate', action='store_true', help='Apply rotation augmentation')
    parser.add_argument('--brightness', action='store_true', help='Apply brightness adjustment')
    parser.add_argument('--cropout', action='store_true', help='Apply cropout augmentation')
    parser.add_argument('--data', type=str, default='./', help='input Path to the dataset')
    parser.add_argument('--output', type=str, default='./output', help='output Path to the dataset')
    parser.add_argument('--display-original', action='store_true', help='Display the original image with bounding boxes')
    parser.add_argument('--position', type=str, default='', choices=['left', 'right', 'top', 'bottom', 'center','random'], help='Position of the cropout')
    parser.add_argument('--box_width_percent', type=int, default=30, help='Width of the cropout box as a percentage of image width (default: 30)')

# Brightness
def adjust_brightness 부분에서 # To edit factor 부분의 factor를 수정해주세요. 
값을 지정해주거나 random factor를 이용하도록 수정해주세요! 
해당 결과물은 이름 뒤에 _br 이 붙습니다.

# Cropout
parser를 통해 상, 하, 좌, 우를 가리고 싶다면 position을 지정해주세요.
random : Bbox 내 상, 하, 좌, 우중 랜덤한 부분 가림.
--position 미사용시 Bbox내 랜던함 부분을 가립니다. 
해당 결과물은 이름 뒤에 _cp 가 붙습니다.

# Rotate
 코드에서 # To edit angle 부분의 angle을 수정해주세요.
 angle을 지정하거나 랜덤한 수치로 적용해주세요 
 해당 결과물은 이름 뒤에 _ro가 붙습니다. 

# merge_multi.py 
준비한 이미지를 png 파일로 변경해주세요. conver_jpg_to_png.py를 사용하면 변환이 가능합니다. 
merge_multi를 사용하기 전 객체의 배경을 제거한 png이미지를 준비해주세요. 
이미지와 라벨 txt를 한 폴더에 모아주세요.
해당 폴어데서 make_list_cur.py를 실행해주세요. -> 이미지와 라벨 파일 경로 텍스트화 

FIX_ME 부분을 수정해주세요.
MIN_MERGE_OBJECTS = 2 : 합성될 객체의 최소 수
MAX_MERGE_OBJECTS = 4 : 합성될 객체의 최대 수
IMAGE_ITER_COUNT = 80 : 객체 수 별 합성할 이미지 수 
IMAGE_HIEGHT = 1387 : 이미지의 높이
IMAGE_WIDTH = 1040 : 이미지의 너비 
INPUT_DIR = os.path.join(os.getcwd(),"split_data/train_original/rm_images/") : make_list_cur.py를 실행한 폴더 ( 배경이 제거된 이미지가 있는 폴더 ) 
OUTPUT_DIR = os.path.join(os.getcwd(),"split_data/merge/") : 합성될 이미지를 저장할 폴더 


# bacK_object.py 
object_img_folder = os.path.join(os.getcwd(),"split_data/merge") : merge_multi.py를 통해 합성된 이미지 파일 경로를 지정해주세요.
background_img_folder = os.path.join(os.getcwd(),"background/images") : 배경으로 사용할 이미지의 폴더 경로를 지정해주세요.
output_folder = os.path.join(os.getcwd(),"split_data/merge_output") : 결과물을 저장할 폴더를 지정해주세요. 

#Bbox_ensure.py 
객체의 Bbox가 이미지 경계 밖에 있는지 검사합니다. 경계 밖에 있다면 경계부분으로 라벨을 수정합니다.
마지막에 실행해주세요. 

folder_path = os.path.join(os.getcwd(),'split_data/merge_output') : back_object.py를 실행한 폴더를 지정해주세요 
image_path = os.path.join(folder_path,'images/train') : image 폴더 경로를 지정해주세요
label_path = os.path.join(folder_path, 'labels/train') : label 폴더 경로를 지정해주세요 

# detect.py 
yolov5를 먼저 clone 해주세요. 
yolov5의 detect.py와 utils/general.py을 변경해주세요 
detect.py 실행 시 왼쪽 상단에 클래스 명, 객체 수가 표시 됩니다. 

sns_send 기능은 비활성화 되어있습니다 사용하고 싶다면 general.py의 마지막 부분인 sns_send를 활성화 해주세요 
import general 부분에 sns_send를 추가해주세요.
sns_send 부분의 """ """ 처리를 삭제해주세요 

# general.py 
plot_object_count 함수가 추가되었습니다.
sms_send 함수가 추가되었습니다. 
sms_send를 사용하고 싶다면 twilio 계정을 생성하고 account를 추가해주세요. 




 
 
