#!/usr/bin/env python
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs
CompressedImage. It converts the CompressedImage into a numpy.ndarray,
then detects and marks features in that image. It finally displays
and publishes the new image - again as CompressedImage topic.
"""
__author__ = 'Simon Haller <simon.haller at uibk.ac.at>'
__version__ = '0.1'
__license__ = 'BSD'
# Python libs
import sys

import cv2
import numpy as np
# OpenCV
# Ros libraries
import rospy
from cv_bridge import CvBridge
from keras.models import model_from_json
# Ros Messages
from sensor_msgs.msg import Image
from std_msgs.msg import String
from keras.preprocessing import image
import tensorflow as tf

# numpy and scipy

# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE = False


class image_feature:

    def __init__(self):
        # subscribed Topic
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw",
                                           Image, self.callback, queue_size=1)
        self.publisher = rospy.Publisher("/custom/emotions", String, queue_size=10)
        self.bridge = CvBridge()
        self.face_cascade = cv2.CascadeClassifier(
            '/home/azban/haarcascade_frontalface_default.xml')
        self.model = model_from_json(open("/home/azban/facial_expression_model_structure.json", "r").read())
        self.model.load_weights('/home/azban/facial_expression_model_weights.h5')  # load weights
        self.graph = tf.get_default_graph()

    def callback(self, ros_data):
        img = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
        # -----------------------------
        # face expression recognizer initialization

        # -----------------------------

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # print(faces) #locations of detected faces

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle to main image

            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            with self.graph.as_default():
                predictions = self.model.predict(img_pixels)  # store probabilities of 7 expressions

            # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])

            emotion = emotions[max_index]
            self.publisher.publish(emotion)


def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"


if __name__ == '__main__':
    main(sys.argv)
