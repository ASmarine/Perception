#!/usr/bin/env python

import tf
import cv2
import yaml
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import Range
from cv_bridge import CvBridgeError
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray

class PerceptionFusion(object):
	def __init__(self):

		# OpenCV Bridge
		self.bridge = CvBridge()

		# Subscribers
		sub1 = rospy.Subscriber('/depth_pcl', PointCloud2, self.depth_pcl_cb) # Point Cloud published by depth_estimator node
		sub2 = rospy.Subscriber('/segmented_img', Image, self.segmented_img_cb) # Segmented image published by object_detector node 
		
		print('Perception Fusion is listening for data..')

		self.pcl_pub = rospy.Publisher('/perception_local_pcl', PointCloud2, queue_size=1) # For testing purposes

		# Load configuration file here
		# config_string = rospy.get_param("/depth_estimator_config")
		# self.config = yaml.load(config_string)

		rospy.init_node('perception_fusion')
		rospy.spin()

	def depth_pcl_cb(self, msg):
		print('Received Point Cloud.')

	def segmented_img_cb(self, msg):
		print('Received Segmented Image.')

		self.segmented_img = msg

		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.segmented_img, "bgr8")
		except CvBridgeError as e:
			print(e)
			print('Error reading Segmented Image.')

		# (rows,cols,channels) = cv_image.shape

		# For debugging purposes
		# cv2.imshow("Image window", cv_image)
		# cv2.waitKey(0)

		self.pcl_pub.publish(PointCloud2()) # For testing purposes
		
		# try:
		# 	self.pcl_pub.publish(PointCloud2('')) # For testing purposes
		# except CvBridgeError as e:
		# 	print(e)


if __name__ == '__main__':
	try:
		PerceptionFusion()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start depth_estimator node.')
