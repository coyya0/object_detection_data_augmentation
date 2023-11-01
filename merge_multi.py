# -*- coding: utf-8 -*-
import random
import math
import numpy as np
from PIL import Image, ImageDraw
import cv2 as cv
import os
################################ FIX_ME ########################################
###                                                                          ###
MIN_MERGE_OBJECTS = 2
MAX_MERGE_OBJECTS = 3
IMAGE_ITER_COUNT = 80
IMAGE_HIEGHT = 1387
IMAGE_WIDTH = 1040
INPUT_DIR = os.path.join(os.getcwd(),"rembg_png_aug/images/") 
OUTPUT_DIR = os.path.join(os.getcwd(),"merge_rembg_png_aug/") 
###                                                                          ###
################################################################################
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
def box_first_point(box):
    '''
    find (x_rate, y_rate) of input box
    Input : [class, x, y, width, height]
    Output : (rate_x, rate_y)
    '''
    rate_x = box[1]-box[3]/2
    rate_y = box[2]-box[4]/2
    return (rate_x, rate_y)

def aD (box):
    '''
    find under_right point of box
    Input : [class, x, y, width, height] (rate)
    Output : (rate_x, rate_y)
    '''
    rate_x = box[1]+box[3]/2
    rate_y = box[2]+box[4]/2
    return (rate_x, rate_y)



class Merger():
    def __init__(self, input_file_path = "", output_file_path = ""):
        
        if input_file_path == "" or output_file_path == "":
            print("Default file path setted")
            ################################ file_path ########################################
            self.input_file_path = INPUT_DIR
            self.output_file_path = OUTPUT_DIR
            ################################ file_path ########################################
         
        ## get image, text file list
        self._merge_image_set = [] 
        self._merge_text_set = [] 
        self.get_file_list()

        self.merge_set_num = [] ## number of merged image in list
        self.merge_set_box = [] ## box of merged image
        self.merge_number = 6 ## the number of merged item
        self.angle = [0, 45, 90, 135, 180, 225, 270, 315,] ## Overlap angle
        self.overlap_distance = [0,20] ## Overlap distance , if 0 : not overlap

        self.image_pos = [] ## the ratio image to move [horizontal, vertical]
        self.image_last_point = (0,0) ## the size of whole merged image 
        ## Merge parameter

    def get_file_list(self):
        '''
        get the list of input image, text file
        this function is done in Constructor of Class 'Merge'
        '''
        image_list = []
        text_list = []
        image_file = open(self.input_file_path + 'target.txt', 'r')
        while True:
            line = image_file.readline()
            if line == "":
                break
            line2 = line[:-1].rsplit('.',1)[0] + ".txt"
            image_list.append(line[:-1]) ## read except '\n'
            text_list.append(line2[:]) ## read exceipt '\n'
        image_file.close()

        self._merge_image_set = image_list
        self._merge_text_set = text_list

        return

    def get_input_file_path(self):
        return self.input_file_path

    def get_output_file_path(self):
        return self.output_file_path

    def text_move(self, move_amount):
        '''
        modify text file information by the moving amount
        '''
        for i in range(self.merge_number):
            self.merge_set_box[i].center_x += move_amount[0]
            self.merge_set_box[i].center_y += move_amount[1]


    def move_image_amount(self):
        '''
        return how much to move image by the ratio of self.image_pos
        return : (horizontal move amount, vertical move amount)
        if image is bigger than screen return (-1, -1)
        '''
        image_last_point = self.image_last_point
        
        ## if merged image is bigger than screen size
        if image_last_point.x >= 1 or image_last_point.y >= 1:
            return (-1, -1)

        hor_move = (1-image_last_point.x)*self.image_pos[0] ## the amount to move horizontally
        ver_move = (1-image_last_point.y)*self.image_pos[1] ## the amount to move vertically
       
        return (hor_move, ver_move)

    def pick_two_index(self, class1 = [], class2 = []):
        '''
        Pick two index which are right class ( one in class1, one in class2 )
        Input : class1, class2 (Array)
        Output : [index1, index2]
        '''
        index_set = []
        if class1 == []:
            idx1 = random.randrange(len(self._merge_image_set))
            index_set.append(idx1)
        else:
            idx_class = -1
            while not idx_class in class1:
                idx = random.randrange(len(self._merge_image_set))
                txt_file = open(self._merge_text_set[idx], "r") 
                txt_line = txt_file.readline()
                txt_line_split = txt_line.split()
                idx_class = int(txt_line_split[0])
                txt_file.close()
            index_set.append(idx)
        if class2==[]:
            idx2 = random.randrange(len(self._merge_image_set))
            index_set.append(idx2)
        else:
            idx_class = -1
            while not idx_class in class2:
                idx = random.randrange(len(self._merge_image_set))
                txt_file = open(self._merge_text_set[idx], "r") 
                txt_line = txt_file.readline()
                txt_line_split = txt_line.split()
                idx_class = int(txt_line_split[0])
                txt_file.close()
            index_set.append(idx)
        return index_set

    def inspect_box(self, index_set = []):
        '''
        Make the box list of merge set 
        Input : index_set (index array)
        Output : self.merge_set_box
        '''
        if index_set == []:
            return -1
        
        for idx in index_set:
            txt_file = open(self._merge_text_set[idx], "r")
            txt_line = txt_file.readline()
            txt_line_split = txt_line.split()
            for i in range(len(txt_line_split)):
                if i==0:
                    txt_line_split[i] = int(txt_line_split[i])
                    continue
                txt_line_split[i] = float(txt_line_split[i])
            self.merge_set_box.append(txt_line_split)
            txt_file.close()

    def bg_transparent(self, input_image):
        '''
        Make image background transparent from black
        Parameter: input_image
        '''
        
        origin_data = input_image.getdata()
        newData = []
        cutoff = 12
        for item in origin_data:
            if item[0] <= cutoff and item[1] <= cutoff and item[2] <= cutoff:
                newData.append((255,255,255,0))
            else:
                newData.append(item)
        input_image.putdata(newData)

    def new_bound(self, fground, bground, fixed_box):
        '''
        make new bounding box of bground
        Parameter : image-foreground, image - background, box_array -  fixed_box
        '''
        cv_fground = cv.cvtColor(np.array(fground), cv.COLOR_RGB2BGR)
        cv_bground = cv.cvtColor(np.array(bground), cv.COLOR_RGB2BGR)
        fground_gray = cv.cvtColor(cv_fground, cv.COLOR_BGR2GRAY)
        bground_gray = cv.cvtColor(cv_bground, cv.COLOR_BGR2GRAY)
        ret_b, bground_mask = cv.threshold(bground_gray, 10, 255, cv.THRESH_BINARY)
        ret, fground_mask = cv.threshold(fground_gray, 20, 255, cv.THRESH_BINARY)
        fground_mask_inv = cv.bitwise_not(fground_mask)
        
        occluded_bground = cv_bground*(fground_mask_inv[:,:,None].astype(cv_bground.dtype))
        occluded_bground = occluded_bground*(bground_mask[:,:,None].astype(occluded_bground.dtype))
        occld_bg_gray = cv.cvtColor(occluded_bground, cv.COLOR_BGR2GRAY)
        ret2, occld_bg_mask = cv.threshold(occld_bg_gray, 20, 255, cv.THRESH_BINARY)
       
        ##kernel = np.ones((5,5),np.uint8) 
        ##kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        ##cv.morphologyEx(occld_bg_mask, cv.MORPH_OPEN, kernel)
        ##cv.morphologyEx(occld_bg_mask, cv.MORPH_CLOSE, kernel)
        ##cv.imwrite('check.png', occld_bg_mask)
        ##cv.imwrite('gray_check.png', occld_bg_gray)

        #contours, hierachy = cv.findContours(occld_bg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        #image, contours, hierachy = cv.findContours(occld_bg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours, hierachy = cv.findContours(occld_bg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        lc_a = 0
        lc = []
        for contour in contours:
            if cv.contourArea(contour) > lc_a:
                lc_a = cv.contourArea(contour)
                lc = contour
        if lc == []:
           fixed_box[1] = []
           return
        x,y,w,h = cv.boundingRect(lc)
        cl = fixed_box[1][0]
        fixed_box[1] = [cl, (x+w/2)/IMAGE_WIDTH, (y+h/2)/IMAGE_HIEGHT, w/IMAGE_WIDTH, h/IMAGE_HIEGHT]
        for i in range(1,len(fixed_box[1])):
            fixed_box[1][i] = round(fixed_box[1][i],6)
        image2 = cv.drawContours(cv_bground, lc, -1, (0,255,0), 2)
        ##cv.imwrite('check2.png', image2)
    
    def find_paste_point(self, angle, overlap_distance, index):
        '''
        Find the target point to paste on
        Input : angle(degree), overlap_distance(%)
        Output : fixed_box(array of fixde box) , point (int, int)
        '''
        standard_item_box = self.merge_set_box[0]
        toadd_item_box = [self.merge_set_box[index][i] for i in range(len(self.merge_set_box[index]))]
        toadd_item_first_point = box_first_point(toadd_item_box)
        if angle in [0, 180]:
            stand_radius = standard_item_box[3]/2
            toadd_radius = toadd_item_box[3]/2
            dest_y = standard_item_box[2]
            if angle == 0:
                dest_x = standard_item_box[1] + stand_radius*(50 - overlap_distance)/50 + toadd_radius
            else:
                dest_x = standard_item_box[1] - stand_radius*(50 - overlap_distance)/50 - toadd_radius

        elif angle in [90, 270]:
            stand_radius = standard_item_box[4]/2
            toadd_radius = toadd_item_box[4]/2
            dest_x = standard_item_box[1]
            if angle == 90:
                dest_y = standard_item_box[2] - stand_radius*(50 - overlap_distance)/50 - toadd_radius
            else:
                dest_y = standard_item_box[2] + stand_radius*(50 - overlap_distance)/50 + toadd_radius
                
        else:
            stand_radius = ((standard_item_box[3])**2+(standard_item_box[4])**2)**0.5/2
            toadd_radius = ((toadd_item_box[3])**2+(toadd_item_box[4])**2)**0.5/2
            if angle == 45 or angle == 315:
                dest_x = standard_item_box[1] + standard_item_box[3]/2*(50 - overlap_distance)/50 + toadd_item_box[3]/2
            else:
                dest_x = standard_item_box[1] - standard_item_box[3]/2*(50 - overlap_distance)/50 - toadd_item_box[3]/2
            if angle == 45 or angle == 135:
                dest_y = standard_item_box[2] - standard_item_box[4]/2*(50 - overlap_distance)/50 - toadd_item_box[4]/2
            else:
                dest_y = standard_item_box[2] + standard_item_box[4]/2*(50 - overlap_distance)/50 + toadd_item_box[4]/2
        
        toadd_item_box[1] = round(dest_x,6)
        toadd_item_box[2] = round(dest_y,6)
        dest_paste_point = box_first_point(toadd_item_box)
        real_destination = (int(IMAGE_WIDTH*(-toadd_item_first_point[0] + dest_paste_point[0])), int(IMAGE_HIEGHT*(-toadd_item_first_point[1] + dest_paste_point[1])))
        print("real_destination : ", real_destination)
        return [standard_item_box, toadd_item_box], real_destination

    def multi_merge(self, classes = [], item_num = 5, num = 1, index = 000):
        '''
        merge many items
        input : classes, item_num = 2, index = 000
        output : save merged image
        '''
        index_set = []
        for iterate in range(num):
            
            if classes == []:
                for i in range(item_num):
                    classes.append([])

            ## select merge_item
            merge_index = []
            for i in range(item_num):
                if classes[i] == []:
                    idx = random.randrange(len(self._merge_image_set))
                    merge_index.append(idx)
                else:
                    idx_class = -1
                    while not idx_class in classes[i]:
                        idx = random.randrange(len(self._merge_image_set))
                        txt_file = open(self._merge_text_set[idx], "r") 
                        txt_line = txt_file.readline()
                        txt_line_split = txt_line.split()
                        idx_class = int(txt_line_split[0])
                        txt_file.close()
                        index_set.append(idx)
            self.inspect_box(merge_index)

            ## select angle, overlap
            angle_list = []
            ovlp_list = []
            for i in range(item_num-1):
                ovlp_list.append(random.choice(self.overlap_distance))
            angle_list = random.sample(self.angle, item_num-1)
            print("angle, overlap = ", angle_list, ovlp_list)
            
            ## fixed_item_box
            fixed_item_box = []
            fixed_item_box.append(self.merge_set_box[0])

            ## background
            background = Image.new("RGBA", (IMAGE_WIDTH, IMAGE_HIEGHT), (0,0,0,0))

            ## first image
        
            input_image = Image.open(self._merge_image_set[merge_index[0]]).convert("RGBA")
            background.paste(input_image, (0,0), input_image)
            
            try:
                for i in range(item_num-1):
                        input_image = Image.open(self._merge_image_set[merge_index[i+1]]).convert("RGBA")
                        before_paste = background.copy()
                        fixed_box, destination = self.find_paste_point(angle_list[i], ovlp_list[i],i+1)
                        fixed_item_box.append(fixed_box[1])
                        shifted_bg = Image.new("RGBA", (IMAGE_WIDTH,IMAGE_HIEGHT), (0,0,0,0))
                        shifted_bg.paste(input_image, destination, input_image)
                        self.new_bound(background, shifted_bg, fixed_box)

                        self.bg_transparent(before_paste)
                        self.bg_transparent(shifted_bg)

                        background.paste(shifted_bg,(0,0), shifted_bg)
                        background.paste(before_paste, (0,0), before_paste)
            except IndexError:
                print("!!!!!!!!!!!!!!!!!!Size Overlap!!!!!!!!!!!!!!!!!!!!!!!!")
                pass
            ## Draw bounding Box
            '''                       
            include_bounding_box = ImageDraw.Draw(background)
            for box in fixed_item_box:
                if box != []:
                    include_bounding_box.rectangle([box_first_point(box)[0]*IMAGE_WIDTH, box_first_point(box)[1]*IMAGE_HIEGHT, (box_last_point(box)[0])*IMAGE_WIDTH, (box_last_point(box)[1])*IMAGE_HIEGHT])
            '''
            

            out_filename = "merged_new_N"+str(item_num)+"_"+str(index)+str(iterate)

            ## save image file
            background = background.convert("RGBA")
            background.save(self.output_file_path + out_filename + ".png", format = "png")
            print(out_filename)

            ## save txt file 
            out_file = open(self.output_file_path + out_filename + ".txt" , 'w')
            for i in range(len(fixed_item_box)):
                for j in range(len(fixed_item_box[i])):
                    out_file.write(str(fixed_item_box[i][j])+" ")
                out_file.write("\n")
            out_file.close()
            
            background.close()
            self.merge_set_num = []
            self.merge_set_box = []



test = Merger()

for i in range(MIN_MERGE_OBJECTS, MAX_MERGE_OBJECTS + 1):
    test.multi_merge([], i, IMAGE_ITER_COUNT)
