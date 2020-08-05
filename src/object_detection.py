#!/usr/bin/env python

import tf
import cv2
import yaml
import rospy
from gate import *
from configuration import *
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridgeError
from matplotlib import pyplot as plt
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray

class ObjectDetector(object):
    def __init__(self):

        # OpenCV Bridge
        self.bridge = CvBridge()

        # Initialize global messages
        self.depth = Image()
        self.odometry = Odometry()
        self.camera_info = CameraInfo()

        # ZED Image subscribers for RGB image
        sub1 = rospy.Subscriber('/camera/zed_node/rgb/image_rect_color', Image, self.zed_rgb_cb)

        #sub2 = rospy.Subscriber('/camera/zed_node/right/image_rect_color', Image, self.zed_right_cb)
        sub3 = rospy.Subscriber('/camera/zed_node/odom', Odometry, self.odom_cb)
        sub4 = rospy.Subscriber('/camera/zed_node/rgb/camera_info', CameraInfo, self.camera_info_cb)
        sub5 = rospy.Subscriber('/camera/zed_node/depth/depth_registered', Image, self.depth_cb)

        # Mono Camera Image Subscriber
        sub = rospy.Subscriber('/mono_image_color', Image, self.mono_cb)
        print('Object Detector listening for images..')

        self.segmented_pub = rospy.Publisher('/segmented_img', Image, queue_size=1)
        self.segmented_pub_2 = rospy.Publisher('/segmented_img_2', Image, queue_size=1)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/camera_info', CameraInfo, queue_size=1)
        self.depth_pub = rospy.Publisher('/depth_registered', Image, queue_size=1)
        self.count = 0

        self.gate_mission = True
        self.path_marker_mission = False
        self.dummy_segment = False # For testing
        
        # Load configuration file here
        # config_string = rospy.get_param("/object_detector_config")
        # self.config = yaml.load(config_string)

        rospy.init_node('object_detector')
        rospy.spin()

    def zed_rgb_cb(self, msg):
        
        print('Received ZED RGB Image.')
        
        self.has_image = True
        self.camera_image = msg
        
        segmented_frame_header = self.camera_image.header # Save header to re-add to published RGB later

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")            
        except CvBridgeError as e:
            print(e)
            print('Error reading ZED RGB image.')
            return

        segmented_frame = cv_image

        if self.gate_mission:
            segmented_frame, segmented_frame_2 = self.find_gate(cv_image)
        elif self.path_marker_mission:
            segmented_frame = self.find_path_marker(cv_image)
        elif self.dummy_segment:
            segmented_frame = self.simple_segment(cv_image)

        try:
            segmented_frame = self.bridge.cv2_to_imgmsg(segmented_frame, "bgr8")
            segmented_frame_2 = self.bridge.cv2_to_imgmsg(segmented_frame_2, "bgr8")
        except Exception as e:
            print(e)
            print('Failed to convert segmented image to Image message!')
            return

        try:
            published_time = rospy.Time.now()
    
            segmented_frame_header.stamp = published_time
            segmented_frame.header = segmented_frame_header
            segmented_frame_2.header = segmented_frame_header
            
            self.depth.header.stamp = published_time            
            self.odometry.header.stamp = published_time
            self.camera_info.header.stamp = published_time
            
            self.depth_pub.publish(self.depth)
            self.odom_pub.publish(self.odometry)
            self.segmented_pub.publish(segmented_frame) 
            self.segmented_pub_2.publish(segmented_frame_2) 
            self.camera_info_pub.publish(self.camera_info)

        except CvBridgeError as e:
            print(e)
            print('Failed to publish ZED data!')

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

    def simple_segment(self, frame):
        image_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
        lower = np.array([87, 156, 60])
        upper = np.array([107, 176, 140])
        
        image_mask = cv2.inRange(image_hsv, lower, upper)   
                
        image_bgr = cv2.cvtColor(image_mask,cv2.COLOR_GRAY2BGR)

        return image_bgr

    def find_gate(self, frame):
        gate_preprocessor = GatePreprocessor()
        gate_classifier=GateClassifier()
        #segmented_frame=gate_preprocessor.get_interest_regions(frame)[1]
        old_frame = np.copy(frame)
        out, out_frame = gate_preprocessor.get_interest_regions(frame)
        segmented_frame= gate_classifier.classify(old_frame, out)
        #print(box)
        #gate_detector=GateDetector()
        #x,y,z,w,segmented_frame=gate_detector.detect(frame)
        return segmented_frame, out_frame

if __name__ == '__main__':
    try:
        ObjectDetector()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detector node.')
