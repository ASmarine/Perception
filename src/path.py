import os
import cv2
import glob
import math
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import broyden1

class PathMarker():

   def __init__(self):
     self.x1 = 0
     self.x2 = 0
     self.y1 = 0
     self.y2 = 0
     self.d = 0
     self.l = 0
     
     self.YELLOW = (100, 255, 200)
     self.RED = (0, 0, 255)

   def RefPoint1(self,frame): 
     # This function returns center point dimensions for a given image
     dimensions = frame.shape
     Y = int(dimensions[0]/2)
     X = int(dimensions[1]/2)
     center_coords = (X,Y)
     return center_coords
   
   def RefPoint2(self,frame, ref_point_1, starting_point, center_point):
     # This function returns the second reference point
     dist = int(math.sqrt((starting_point[0]-center_point[0])**2 + (starting_point[1]-center_point[1])**2))
     ref_point = (ref_point_1[0], ref_point_1[1]+dist)
     return ref_point
   
   # This function draws a point at the given coordinates
   def DrawPoint1(self,frame, text, color, X, Y):
  	 	font = cv2.FONT_HERSHEY_SIMPLEX
   		coord = (X, Y)
   		text_coord = (X+25, Y+25)
   		fontScale = 3
   		textScale = 1
   		fontColor = color
   		lineType = 7
   		cv2.putText(frame, '.', coord, font, fontScale, fontColor, lineType)
   		cv2.putText(frame, text, text_coord, font, textScale, fontColor, 6)
   		return frame
   # This function draws a line between two given points
   def DrawLine(self,frame, point1, point2):
     cv2.line(frame, point1, point2, (0, 255, 0), thickness=3, lineType=8)
     return frame
   def PreProcess(self,frame):
     imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     # apply gaussian blur
     blur = cv2.GaussianBlur(frame,(5,5),0)
     blur1 = cv2.GaussianBlur(blur,(5,5),0)
     blur2 = cv2.GaussianBlur(blur1,(5,5),0)
     blur3 = cv2.GaussianBlur(blur2,(5,5),0)
     blur4 = cv2.GaussianBlur(blur3,(5,5),0)

     # convert from bgr to hsv
     hsv = cv2.cvtColor(blur4, cv2.COLOR_BGR2HSV)

     # define mask boundaries
     lower_blue = np.array([0,100,20])
     upper_blue = np.array([100,255,255])   
     
     # apply HSV mask
     err = cv2.inRange(hsv, lower_blue, upper_blue)
     
     # apply morphological operators
     _, mask = cv2.threshold(err, 250, 255, cv2.THRESH_BINARY_INV)
     kernel = np.ones((5, 5), np.uint8)
     dilation = cv2.dilate(mask, kernel)

     # apply canny line detection
     edges = cv2.Canny(dilation,1000,1200)
     edges = np.float32(edges)
     return  edges

   def CornerDetect(self,img, edges):
     ret_list_x = []
     ret_list_y = []
     
     # GFTT Params
     max_num_of_corners = 18
     quality_level = 0.05
     min_distance = 34
     
     corners = cv2.goodFeaturesToTrack(edges, max_num_of_corners, quality_level, min_distance)
     corners = np.int0(corners)
     
     for i in corners:
       x1, y1 = i.ravel()
       cv2.circle(img,(x1,y1),3,255,-1)
       ret_list_x.append(x1)
       ret_list_y.append(y1)
     return img, ret_list_x, ret_list_y, corners
   # This function returns the starting point and the ending point
   # TODO: Remember to check if the shape is vertical or horizontal  
   def get_starting_and_ending_points(self,corners):
     corners = corners[corners[:,0,1].argsort()]

     range = 2
     x = [p[0][0] for p in corners[0:range]]
     y = [p[0][1] for p in corners[0:range]]
     
     end_centroid = (int(sum(x) / range), int(sum(y) / range))

     x = [p[0][0] for p in corners[-range:]]
     y = [p[0][1] for p in corners[-range:]]

     start_centroid = (int(sum(x) / range), int(sum(y) / range))

     return start_centroid, end_centroid

   def f1(self,x):
     return (math.sqrt( abs( (self.l**2) - (self.x1**2) + (2*self.x1*x) - (x**2)) ) + self.y1 )

   def f2(self,x):
     return (math.sqrt( abs( (self.l**2) - (self.x2**2) + (2*self.x2*x) - (x**2)) ) + self.y2 )

   def twofuncs(self,x):
     y = [ self.f1(x[0])-x[1], self.f2(x[0])-x[1] ]
     return y

   def eqs(self,p):
     x, y = p
     return (y - (math.sqrt(abs((self.l**2) - (self.x1**2) + (2*self.x1*x) - (x**2))) + self.y1), y - (math.sqrt(abs((self.l**2) - (self.x2**2) + (2*self.x2*x) - (x**2))) + self.y2))
   
   def get_center_point(self,corners, starting_point, ending_point):
     #global x1, y1, x2, y2, l
     self.x1 = starting_point[0]
     self.y1 = starting_point[1]
     self.x2 = ending_point[0]
     self.y2 = ending_point[1]
     self.d = int(math.sqrt((self.x1-self.x2)**2+(self.y1-self.y2)**2))
     self.l = int((self.d/math.sin(math.radians(135)))*math.sin(math.radians(22.5)))
     print('({},{}) ({},{})\nbase = {} side = {}'.format(self.x1,self.y1,self.x2,self.y2,self.d,self.l))
     #sol = broyden1(twofuncs, [0,0], False)
     #sol = fsolvee(eqs, (5,5))
     # sol = fsolve(self.eqs, (self.x1+0.1,self.y1-0.1))
     # print(sol)
     # center_point = (, int(sol[1]))
     return (self.x1, self.y1)
   
   def find_path(self,img):
     im = np.array(img)
  
     # Draw reference point
     ref_coords = self.RefPoint1(img) # get reference point coordinates
     im = self.DrawPoint1(im, 'Reference Point', self.RED, ref_coords[0], ref_coords[1]) # draw

     # Preprocess image
     edges = self.PreProcess(img)
     edges_w_corners, edges_x_l, edges_y_l, corners = self.CornerDetect(img, edges)
     # 
     # Get marker starting and ending points and draw them
     starting_point, ending_point = self.get_starting_and_ending_points(corners)
     im = self.DrawPoint1(im, 'Starting Point', self.YELLOW, starting_point[0], starting_point[1]) 
     im = self.DrawPoint1(im, 'Ending Point', self.YELLOW, ending_point[0], ending_point[1])
     
   #   # Get the center corner point and draw it
     center_point = self.get_center_point(corners, starting_point, ending_point)
     im = self.DrawPoint1(im, 'Center Point', self.YELLOW, center_point[0], center_point[1]) 
     
     # Connect the points with lines
     im = self.DrawLine(im, starting_point, center_point)
     im = self.DrawLine(im, center_point, ending_point)
     
     # Calculate and draw second reference point
     second_ref_coords = self.RefPoint2(im, ref_coords, starting_point, center_point) # get second reference point coordinates
     im = self.DrawPoint1(im, 'Reference Point 2', self.RED, second_ref_coords[0], second_ref_coords[1]) # draw
     
     im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
     return im_rgb

path = PathMarker()

frame = cv2.imread("/home/zeyad/Path4.png",1)
processed_frame = path.find_path(frame)
cv2.imshow('',processed_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
