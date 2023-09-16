import os
import random
import glob
from os import getcwd
import cv2
import numpy as np
     
current_dir = getcwd()+'/'

#FIXME
#in_filename = ['apple2pringles.mp4','banana2cass.mp4']

in_filename = [current_dir+"/"+'10.mp4']
out_filename = "videosplit"
out_dirname = "image2"

cnt = 1000


for vid in in_filename:
	print("Reading "+vid)
	cap = cv2.VideoCapture(vid)
	if(cnt*10000==0):
		print("Reading... %d\n",cnt)

	while(True):
		ret, frame = cap.read()
		print(ret)
		if ret:
			cv2.imwrite(out_dirname+"/"+out_filename+"_"+str(cnt)+".jpg",frame)
			cnt += 1
		else:
			print("File grab failed")
			break

	cap.release()
