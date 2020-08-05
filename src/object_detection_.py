#!/usr/bin/env python
 	
import tf
import cv2
import yaml
import rospy
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridgeError
from std_msgs.msg import Float32MultiArray

class ObjectDetector(object):
	def __init__(self):

		# OpenCV Bridge
		self.bridge = CvBridge()

		# Initialize global messages
		self.odometry = Odometry()
		self.camera_info = CameraInfo()
		self.depth = Image()

		# ZED Image subscribers for left and right channels
		sub1 = rospy.Subscriber('/camera/zed_node/rgb/image_rect_color', Image, self.zed_left_cb)
		#sub2 = rospy.Subscriber('/camera/zed_node/right/image_rect_color', Image, self.zed_right_cb)
		sub3 = rospy.Subscriber('/camera/zed_node/odom', Odometry, self.odom_cb)
		sub4 = rospy.Subscriber('/camera/zed_node/rgb/camera_info', CameraInfo, self.camera_info_cb)
		sub5 = rospy.Subscriber('/camera/zed_node/depth/depth_registered', Image, self.depth_cb)

		# Mono Camera Image Subscriber
		sub = rospy.Subscriber('/mono_image_color', Image, self.mono_cb)
		print('Object Detector listening for images..')

		self.segmented_pub = rospy.Publisher('/segmented_img', Image, queue_size=1)
		self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
		self.camera_info_pub = rospy.Publisher('/camera_info', CameraInfo, queue_size=1)
		self.depth_pub = rospy.Publisher('/depth_registered', Image, queue_size=1)

		self.count = 0
		
		# Load configuration file here
		# config_string = rospy.get_param("/object_detector_config")
		# self.config = yaml.load(config_string)

		rospy.init_node('object_detector')
		rospy.spin()

	def zed_left_cb(self, msg):
		print('Received ZED Left Image.')
		self.has_image = True
		self.camera_image = msg
		
		segmented_img_header = self.camera_image.header
		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
		except CvBridgeError as e:
			print(e)
			print('Error reading ZED left image.')

		# (rows,cols,channels) = cv_image.shape

		# For debugging purposes
		# cv2.imshow("Image window", cv_image)
		# cv2.waitKey(0)

		cv_image_mask = self.simple_segment(cv_image) # Segment the image based on color
		
		try:
			published_time = rospy.Time.now()
			segmented_img = self.bridge.cv2_to_imgmsg(cv_image_mask, "bgr8")
			
			segmented_img_header.stamp = published_time
			segmented_img.header = segmented_img_header
			
			self.odometry.header.stamp = published_time
			self.camera_info.header.stamp = published_time
			self.depth.header.stamp = published_time			
			
			self.segmented_pub.publish(segmented_img) # For testing purposes
			self.odom_pub.publish(self.odometry)
			self.camera_info_pub.publish(self.camera_info)
			self.depth_pub.publish(self.depth)

			# self.segmented_pub.publish(Image()) # For testing purposes
		except CvBridgeError as e:
			print(e)

	def zed_right_cb(self, msg):
		print('Received ZED Right Image.')
		self.has_image = True
		self.camera_image = msg

		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
		except CvBridgeError as e:
			print(e)
			print('Error reading ZED left image.')

		# (rows,cols,channels) = cv_image.shape

		# For debugging purposes
		# cv2.imshow("Image window", cv_image)
		# cv2.waitKey(0)
		
		try:
			self.segmented_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8")) # For testing purposes
			# self.segmented_pub.publish(Image()) # For testing purposes
		except CvBridgeError as e:
			print(e)
			print('Error reading ZED right image.')

		# (rows,cols,channels) = cv_image.shape

		# For debugging purposes
		# cv2.imshow("Image window", cv_image)
		# cv2.waitKey(0)

		try:
			# self.test_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8")) # For testing purposes
			# self.segmented_pub.publish(Image()) # For testing purposes
			pass
		except CvBridgeError as e:
			print(e)

	def mono_cb(self, msg):
		print('Received Mono Image.')
		self.has_image = True
		self.camera_image = msg

		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
		except CvBridgeError as e:
			print(e)
			print('Error reading Mono Camera image.')

		# (rows,cols,channels) = cv_image.shape

		# For debugging purposes
		# cv2.imshow("Image window", cv_image)
		# cv2.waitKey(0)

		try:
			# self.test_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8")) # For testing purposes
			self.segmented_pub.publish(Image()) # For testing purposes
		except CvBridgeError as e:
			print(e)

	def odom_cb(self, msg):
		print('Received Odometry.')
		self.odometry = msg

	def camera_info_cb(self, msg):
		print('Received Camera Info.')
		self.camera_info = msg

	def depth_cb(self, msg):
		print('Received Depth Image.')
		self.depth = msg

	def simple_segment(self, image_src):

		image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
	
		lower = np.array([87, 156, 60])
		upper = np.array([107, 176, 140])
		
		image_mask = cv2.inRange(image_hsv, lower, upper)	
				
		image_bgr = cv2.cvtColor(image_mask,cv2.COLOR_GRAY2BGR)

		return image_bgr


class GateDetector:
	def __init__():
		pass

# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
	pixel = image_hsv[y,x]

	#you might want to adjust the ranges(+-10, etc):
	upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
	lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
	print(pixel, lower, upper)

	image_mask = cv2.inRange(image_hsv,lower,upper)
	print(lower, upper)
	cv2.imshow("mask",image_mask)


if __name__ == '__main__':
	try:
		image_hsv = None   # global ;(
		pixel = (20,60,80) # some stupid default

		ObjectDetector()

	except rospy.ROSInterruptException:
		rospy.logerr('Could not start object detector node.')
