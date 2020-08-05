#!/usr/bin/env python

import cv2
import yaml
import pickle
import numpy as np
import pandas as pd
from configuration import *
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
from utils import * 
import time

class GatePreprocessor:
    def __init__(self):
        #our_values
        self.lower_hsv = np.array([35, 100, 20], 'uint8')   # Our own color threshold 
        self.upper_hsv = np.array([100, 255, 255], 'uint8') # Our own color threshold
        self.use_hsv = True
        self.min_cont_size = 37 # min contours size
        self.max_cont_size = 92 # max contours size
        self.roi_size = 1000 # box size
        self.morph_ops = True # testing

        ''' KERNEL '''
        self.kernel_dil = np.ones( (5, 5), np.uint8) # basic filter
        #self.kernel = conkernel_diag_pos
        self.shapes = {1: "vertical", 2: "horizontal", 3: "square"} # so we can change names quicker
        self.shape_buffer = 15
        self.frame_size = (744, 480)
        self.shape_ratio_lower = 0.20
        self.shape_ratio_upper = 1.80

    def preprocess(self,img):
        if self.use_hsv:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img, self.lower_hsv, self.upper_hsv)
            output = cv2.bitwise_and(img, img, mask=mask)
        return output, mask

    def color_subtract(self, frame):
      blue = frame.copy()
      green = frame.copy()
      red = frame.copy()

      blue[:, :, 0] = 255
      green[:, :, 1] = 255
      red[:, :, 2] = 255

      blue_gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
      green_gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
      red_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

      green_blue = green_gray - blue_gray

      return green_blue
    
    def filter_contours(self, frame_contours):
        new_cont_list = []
        for cont in frame_contours:
            cont_len = len(cont)
            if ( (cont_len > self.min_cont_size) and (cont_len < self.max_cont_size) ):
                new_cont_list.append(cont)
        filtered_contours = np.array(new_cont_list)
        return filtered_contours

    def create_dataset(self, contours):
        if contours is None:
            return None, None
        X = []
        y = []

        for cont in contours:
            if cont is not None:
                M = cv2.moments(cont)
                #cy = int(M["m01"] / M["m00"] + 1) # add 1 to avoid division by zero
                cy = int(M["m01"] / M["m00"] + 1) # add 1 to avoid division by zero
                perimeter = cv2.arcLength(cont, True)
                X.append([perimeter, cy])

        return (pd.DataFrame(X), None )

    def nearest_neighbors(self, dataset, distance=False):
        if dataset is None:
            return None
        elif len(dataset) < 2:
            return None
        else:
            nn = NearestNeighbors(n_neighbors=2)
            print('Dataset before fitting', dataset)
            nn.fit(dataset)
            print('Dataset after fitting', dataset)
            dist_ind, nn.kneighbors(dataset, return_distance=distance)
            print('Dataset after knn', dataset)
            print('Dist', dist)
            print('Ind', ind)
        return 

    def create_pairs(self, conts):
        if conts is None:
            return None
        new_list = []
        for i in conts:
            tmp_list = []
            tmp_list.append(i[0])
            tmp_list.append(i[1])
            new_list.append(tmp_list)
        return new_list

    def return_box_pairs(self, filtered_contours, converted_pairs):
        if converted_pairs is None:
            return None
        pair_tuples = []
        counter = 0
        for pair in converted_pairs:
            first = filtered_contours[pair[0]]
            second = filtered_contours[pair[1]]
            first_box = cv2.boundingRect(first)
            second_box = cv2.boundingRect(second)
            pair_tuples.append((first_box, second_box))
        return pair_tuples

    def detect_whole_gate(self, interest_regions, shape):
        ret = []

        if interest_regions:

            area_max = 0
            area_max2 = 0
            len_rois = len(interest_regions)
            print("length_rois:",len_rois)

            if len_rois > 0:

                counted_rois = None
                for i in range(0, len_rois):
                    area_mult = 0.4

                    cx, cy, cw, ch = interest_regions[i][0]
                    cx2, cy2, cw2, ch2 = interest_regions[i][1]

                    carea = float(cw) * float(ch)
                    neighbor_area_buffer = carea * area_mult
                    area_check_upper = carea + neighbor_area_buffer
                    area_check_lower = carea - neighbor_area_buffer
                    carea2 = float(cw2) * float(ch2)
                    neighbor_area_buffer2 = carea2 * area_mult
                    area_check_upper2 = carea2 + neighbor_area_buffer2
                    area_check_lower2 = carea2 - neighbor_area_buffer2

                    # max_carea = max(carea, carea2)

                    if (area_check_lower2 <= (cw*ch) <= area_check_upper2) and self.get_shape(interest_regions[i][0]) == shape and self.get_shape(interest_regions[i][1]) == shape:
                        # neighbor_count += 1
                        counted_rois = interest_regions[0]

                        min_x = self.frame_size[0]
                        min_y = self.frame_size[1]
                        max_x = 0
                        max_y = 0
                        max_w = 0
                        max_h = 0

                        for cr in counted_rois:
                            if min_x > cr[0]:
                                min_x = cr[0]
                            if min_y > cr[1]:
                                min_y = cr[1]
                            if max_x + max_w < cr[0] + cr[2]:
                                max_x = cr[0]
                                max_w = cr[2]
                            if max_y + max_h < cr[1] + cr[3]:
                                max_y = cr[1]
                                max_h = cr[3]

                        if counted_rois is not None:
                            w_ret = max_x - min_x + max_w
                            h_ret = max_y - min_y + max_h

                            ret.append((min_x, min_y, w_ret, h_ret))
                 #if len(ret) > 0:
                    #return ret

        return ret

    def find_pips(self, img):
    # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create()
        # Detect blobs.
        keypoints = detector.detect(img)

        return keypoints

    def get_crop_from_bounding_box(self, img, box):
        x, y, w, h = box
        return img[y:y+h, x:x+w]

    def find_number_of_pips(self, box, img):
        crop = self.get_crop_from_bounding_box(img, box)
        print("get_crop_from_bounding_box:",crop )
        return len(self.find_pips(crop))

    def get_shape(self, roi, ratio_lower = None, ratio_upper = None):
        if roi == None:
            return None
        else:
            x, y, w, h = roi
            if w == 0 or h == 0:
                return None

        if ratio_lower is None:
            ratio_lower = self.shape_ratio_lower
        if ratio_upper is None:
            ratio_upper = self.shape_ratio_upper

        #if ( (h >= (w + buff) ) or (h >= (w - buff) )):
        if float(w)/float(h) < ratio_lower:
            return self.shapes[1] # vertical
        #elif ( (h <= (w + buff) ) or (h <= (w - buff) )):
        elif float(w)/float(h) > ratio_upper:
            return self.shapes[2] # horizontal
        else:
            return self.shapes[3] # square
  
    def get_interest_regions(self, frame):

        color_filt_frame, mask = self.preprocess(frame) # color filtering
        erosion = cv2.erode(mask,kernel,iterations = 1)
        dilated = cv2.dilate(erosion,kernel,iterations = 1)
        #plt.imshow(dilated)

        frame_c, frame_contours, frame_heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        original_frame = np.copy(frame)
        max_len=0
        min_len = 99999999
        for cont in frame_contours:
          max_len = max(max_len, len(cont))
          min_len = min(min_len, len(cont))
        cv2.drawContours(frame,frame_contours,-1,(0,0,255),3)
        
        filtered_contours = self.filter_contours(frame_contours) # filter the contours based on size
        
        cv2.drawContours(original_frame,filtered_contours,-1,(0,0,255),3)

        X_df, y_df = self.create_dataset(filtered_contours) # not using y_df
        contour_pairs = self.nearest_neighbors(X_df)
        print("contour_pairs:",contour_pairs)

        converted_pairs = self.create_pairs(contour_pairs)

        roi_pairs = self.return_box_pairs(filtered_contours, converted_pairs)

        boxes = self.detect_whole_gate(roi_pairs, self.shapes[1])
        #print('boxes:',boxes)

        boxes = [cv2.boundingRect(c) for c in filtered_contours] # make boxes around contours
        interest_regions = [b for b in boxes if b[2]*b[3] > self.roi_size and self.find_number_of_pips(b, frame)>0]
        #print('int_regions:',interest_regions)
        return interest_regions, frame

class GateClassifier:
    def __init__(self):
        # self.lvsm = pickle.load(open("/home/perihane_youssef/Downloads/model_(1).pkl"))
        self.lsvm = joblib.load("/home/perihane_youssef/catkin_ws/src/ASURT-AUV-19-ROSPKG/src/perception/src/model_.pkl")
        #self.print("\nLoading Gate model from disk...\n")
        self.dims = (80, 80)
        self.bins = 9
        self.cell_size = (8, 8)
        self.block_stride = (8, 8)
        self.block_size = (16, 16)
        self.hog = cv2.HOGDescriptor(
                self.dims,
                self.block_size,
                self.block_stride,
                self.cell_size,
                self.bins)
        self.min_prob = .6
        

    def classify(self, frame, roi):
        gate = None
        max_val = 0
        max_box=0
        print('roi', roi)
        for box in roi:
            x, y, w, h = box
            window = frame[y:y + h, x:x + w, :]
            window_resized = cv2.resize(window, self.dims)
            feat = self.hog.compute(window_resized)
            feat_reshape = feat.reshape(1, -1)
            prob = self.lsvm.predict_proba(feat_reshape)[0]
            print(prob)
            # prediction = self.lsvm.predict(feat_reshape)
            gate_class = prob[1] # corresponds to class 1 (positive gate)
            if (gate_class > self.min_prob and gate_class > max_val):
                max_val=gate_class
                gate=box
        cv2.drawContours(frame,gate,-1,(0,0,255),3)
        return frame



class GateDetector:

    def __init__(self):
        self.classifier = GateClassifier()
        self.found =  False
        self.preprocess = GatePreprocessor()
        self.directions = [0,0]
        self.isTaskComplete = False
        self.shapes = {1: "vertical", 2: "horizontal", 3: "square"} # so we can change names quicker
        self.shape_buffer = 15
        self.shape_list = []
        self.is_direction_center = True
        self.is_red_left = False
        self.frame_size = (744, 480)

    # takes a single-roi coordinate as (x, y, w, h) and a buffer as an int
    # returns the shape as a string
    def get_shape(self, roi, buff):
        if roi == None:
            return None
        else:
            x, y, w, h = roi

        #if ( (h >= (w + buff) ) or (h >= (w - buff) )):
        if h - w > buff:
            return self.shapes[1] # vertical
        #elif ( (h <= (w + buff) ) or (h <= (w - buff) )):
        elif w - h > buff:
            return self.shapes[2] # horizontal
        else:
            return self.shapes[3] # square

    ## MARK's DICE METHOD
    def find_pips(self, img):
    # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create()
        # Detect blobs.
        keypoints = detector.detect(img)
        
        return keypoints

    ## MARK's helper
    def get_crop_from_bounding_box(self, img, box):
        x, y, w, h = box
        return img[y:y+h, x:x+w]

    # now returns (found, directions, shape-of-roi, size)
    def detect(self, frame):
        if frame is not None:
            height, width, ch = frame.shape
            center = (width / 2, height / 2)
            regions_of_interest = self.preprocess.get_interest_regions(frame)
            print("you are here:",regions_of_interest)

            for x, y, w, h in regions_of_interest:
                cv2.drawContours(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


            gate = self.classifier.classify(frame, regions_of_interest)


            gate_shape = self.get_shape(gate, self.shape_buffer)

            if gate_shape == self.shapes[3] or gate_shape == self.shapes[1]:
                gate = None

            if (gate == None):
                self.directions = [0, 0]
                self.found = False
                gate_shape = None
                w, h = 0, 0
            else:
                x, y, w, h = gate
                cv2.rectangle(frame, (x, y), (x + w, y + h), utils.colors["blue"], 6)

                w_pad = w / 7
                h_pad = h / 7
                if self.is_direction_center:
                    self.directions = utils.get_directions(center, x, y, w, h)
                    cv2.rectangle(frame, (x + (3*w_pad), y + (3*h_pad)), (x + (4 * w_pad), y + (4 * h_pad)), utils.colors["green"], 2)
                else:
                    if self.is_red_left:
                        self.directions = utils.get_directions_left(center, x, y, w, h)
                        cv2.rectangle(frame, (x + (2*w_pad), y + (3*h_pad)), (x + (3 * w_pad), y + (4 * h_pad)), utils.colors["green"], 2)
                    else:
                        self.directions = utils.get_directions_right(center, x, y, w, h)
                        cv2.rectangle(frame, (x + (4*w_pad), y + (3*h_pad)), (x + (5 * w_pad), y + (4 * h_pad)), utils.colors["green"], 2)
                self.found = True
            return (self.found, self.directions, gate_shape, (w, h),frame)
        else:
            print('error no frame')
            return False, None, None, None


