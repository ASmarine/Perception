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

class DepthEstimation(object):
	def __init__(self):

		# OpenCV Bridge
		self.bridge = CvBridge()

		# Subscribers for ZED data
		sub1 = rospy.Subscriber('/zed/depth/depth_registered', Image, self.zed_depth_cb) # Depth map image registered on left image (32-bit float in meters by default)
		sub2 = rospy.Subscriber('/zed/point_cloud/cloud_registered', PointCloud2, self.zed_cloud_cb) # Registered color point cloud
		sub3 = rospy.Subscriber('/zed/confidence/confidence_map', Image, self.zed_confidence_map_cb) # Confidence image

		# Sonar Data Subscriber
		sub4 = rospy.Subscriber('/sonar_range', Range, self.sonar_cb)
		
		print('Depth Estimator is listening for data..')

		self.pcl_pub = rospy.Publisher('/depth_pcl', PointCloud2, queue_size=1) # For testing purposes

		# Load configuration file here
		# config_string = rospy.get_param("/depth_estimator_config")
		# self.config = yaml.load(config_string)

		rospy.init_node('depth_estimator')
		rospy.spin()

	def zed_depth_cb(self, msg):
		print('Received ZED depth Image.')
		# Update point cloud based on new input here
		# Then publish
		self.update_and_publish_pcl()

	def zed_cloud_cb(self, msg):
		print('Received ZED Point Cloud.')
		# Update point cloud based on new input here
		# Then publish
		self.update_and_publish_pcl()

	def zed_confidence_map_cb(self, msg):
		print('Received ZED Confidence Map.')
		# Update point cloud based on new input here
		# Then publish
		self.update_and_publish_pcl()

	def sonar_cb(self, msg):
		print('Received Sonar range data.')
		# Update point cloud based on new input here
		# Then publish
		self.update_and_publish_pcl()

	def update_and_publish_pcl(self):
		self.pcl_pub.publish(PointCloud2())

if __name__ == '__main__':
	try:
		DepthEstimation()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start depth_estimator node.')
