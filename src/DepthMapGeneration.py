#!/usr/bin/env python

import cv2
import yaml
import rospy
import time
import numpy as np
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Imu
from cv_bridge import CvBridgeError
from std_msgs.msg import Float32MultiArray
from matplotlib import pyplot as plt


class DepthMapGeneration(object):
	def __init__(self):
		
		global leftimg
		global rightimg
		
		leftimg=0
		rightimg=0

		# OpenCV Bridge
		self.bridge = CvBridge()
		rospy.init_node('DepthMapGeneration')

		# Subscribers
		sub1 = rospy.Subscriber('rexrov/rexrov/cameraleft/camera_image',numpy_msg(Image), self.leftimg_cb) 
		sub2 = rospy.Subscriber('rexrov/rexrov/cameraright/camera_image', numpy_msg(Image), self.rightimg_cb)
		sub3 = rospy.Subscriber('rexrov/rexrov/cameraleft/camera_info', CameraInfo, self.left_camera_info_cb)
		sub4 = rospy.Subscriber('rexrov/imu', Imu, self.imu_cb)

		self.pub = rospy.Publisher('/stereo/depth_map', Image)
		self.pub1 = rospy.Publisher('/stereo/camera_info_modified', CameraInfo)
		self.pub2 = rospy.Publisher('/stereo/left_image_raw_modified', Image)
		self.pub3 = rospy.Publisher('/imu/updated', Imu)

		print('Waiting for Left and Right Images')

		self.right_image_raw = Image()
		self.left_image_raw = Image()
		self.left_camera_info = CameraInfo()
		self.imu= Imu()

		rospy.spin()

	def leftimg_cb(self, msg):
		print('Received Left Image.')
		self.left_image_raw = msg
		# global leftimg
				
		# leftimg=msg
		# try:
		# 	leftimg = self.bridge.imgmsg_to_cv2(leftimg, "bgr8")
		# except CvBridgeError as e:
		# 	print(e)
		# 	print('Error reading Left Image')
		# if leftimg == None: 
  #   		    raise Exception("could not load image !")

	def rightimg_cb(self, msg):

		print('Received Right Image')

		self.right_image_raw = msg

		#cv2.imshow('right', rightimg)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		# if self.right_image_raw == None: 
		# 	raise Exception("could not load image !")

		try:
			rightimg = self.bridge.imgmsg_to_cv2(self.right_image_raw, "bgr8")
		except CvBridgeError as e:
			print(e)
			print('Error reading Right Image')
			return

		try:
			leftimg = self.bridge.imgmsg_to_cv2(self.left_image_raw, "bgr8")
		except CvBridgeError as e:
			print(e)
			print('Error reading Left Image')
			return

		# if leftimg == None: 
  #   		    raise Exception("could not load image !")

		#rightt = cv2.imwrite('/home/stereoright.png',rightimg)
 
		#cv2.imshow('right', rightimg)
		#leftimg1 = np.array(leftimg, dtype=np.uint8)
		#rightimg1 = np.array(rightimg, dtype=np.uint8)

		left=cv2.cvtColor(leftimg, cv2.COLOR_BGR2GRAY)
		right=cv2.cvtColor(rightimg, cv2.COLOR_BGR2GRAY)

		stereo=cv2.StereoBM_create(numDisparities=32, blockSize=21)

		#cv2.imshow('right', right)
		
		disparity = stereo.compute(left,right)
		disparity=~disparity
		
		#cv2.imshow('disparity', disparity)		
		
		try:
			published_time = rospy.Time.now()
			
			disparity = np.array(disparity, dtype=np.uint16)
			# depth_image_msg = cv2.cvtColor(disparity, cv2.COLOR_GRAY2BGR565)
			depth_image_msg = self.bridge.cv2_to_imgmsg(disparity)
			depth_image_msg.header = self.left_image_raw.header
			
			depth_image_msg.header.stamp = published_time
			self.left_image_raw.header.stamp = published_time
			self.left_camera_info.header.stamp = published_time
			self.imu.header=self.left_image_raw.header
			self.imu.header.stamp = published_time

			self.pub.publish(depth_image_msg)
			self.pub1.publish(self.left_camera_info)
			self.pub2.publish(self.left_image_raw)
			self.pub3.publish(self.imu)

		except CvBridgeError as e:
			print(e)
		
		#plt.imshow(disparity,'gray')
		#plt.show()
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	def left_camera_info_cb(self, msg):
		self.left_camera_info = msg

	def imu_cb(self, msg):
		self.imu = msg

if __name__ == '__main__':
	try:
		DepthMapGeneration()
	except rospy.ROSInterruptException:
		rospy.logerr('Error')
