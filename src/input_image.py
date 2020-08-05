#!/usr/bin/env python 
import tf
import cv2
import yaml
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridgeError
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt

class PhotoPublisher(object):
    def __init__(self):

        # OpenCV Bridge
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/camera/zed_node/rgb/image_rect_color', Image, queue_size=1) 
        rospy.init_node('photo_publisher')
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture('/home/perihane_youssef/catkin_ws/src/GOPR1142.MP4')
         
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
          print("Error opening video stream or file")
         
        # Read until video is completed
        while(cap.isOpened()):
            
          # Capture frame-by-frame
          ret, frame = cap.read()
          if ret == True:
         
           # Display the resulting frame
           # ZED Image subscribers for left and right channels
           self.pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8")) 
           print("Published!")

        # When everything done, release the video capture object
        cap.release()
         
        # Closes all the frames
        cv2.destroyAllWindows()
        rospy.spin()
        
if __name__ == '__main__':
    try:
        print("it worked")
        PhotoPublisher()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detector node.')

