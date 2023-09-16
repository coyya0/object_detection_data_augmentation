#!/usr/bin/python
from PIL import Image
import os
import shutil

#TODO: path name , resize_data
BASE_DIR = os.getcwd()
path =BASE_DIR + '/original_img/'
dirs = os.listdir(path)

resize_data = [
    {"folder_name" : "img_resize", "width" :640, "length" : 480}
]

for d in resize_data:
    # Delete the existing folder if it exists
    if os.path.isdir(d['folder_name']):
        shutil.rmtree(d['folder_name'])

    # create folder
    os.mkdir(d['folder_name'])

    # img resize
    for item in dirs:
        if os.path.isfile(path+item):
            print(path+item)
            im = Image.open(path+item)
            f, e = os.path.splitext(d['folder_name'] + "/" +item)
            imResize = im.resize((d['width'], d['length']), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)
