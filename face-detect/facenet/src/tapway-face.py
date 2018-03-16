#   import facenet libraires
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import os
import align.detect_face

#  import other libraries
import cv2
import matplotlib.pyplot as plt
import time ## facenet/contributed
import sys ### oka import system
import numpy as np ## for to_rgb

#   setup facenet parameters
gpu_memory_fraction = 1.0
#minsize = 50 # minimum size of face
minsize = 20#oka change
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

# def add_overlays(frame, faces, frame_rate):
#     if faces is not None:
#         for face in faces:
#             face_bb = face.bounding_box.astype(int)
#             cv2.rectangle(frame,
#                           (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
#                           (0, 255, 0), 2)
#             if face.name is not None:
#                 cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
#                             thickness=2, lineType=2)

#     cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
#                 thickness=2, lineType=2)

#   Start code from facenet/src/compare.py
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
        log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(
            sess, None)
#   end code from facenet/src/compare.py
    
    frame_interval = 3
    fps_display_interval = 10
    frame_rate = 0
    frame_count = 0

    #file = "http://192.168.0.4/video/mjpg.cgi"
    file = 0
    camera = cv2.VideoCapture(file)
    cv2.namedWindow('',0)
    _,img = camera.read()
    height,width,_ = img.shape
    cv2.resizeWindow('',width,height)
    
    start_time = time.time()

    while True:
        _, img = camera.read()


        if (frame_count%frame_interval) == 0:
            print("Frame: "+str(camera.get(cv2.CAP_PROP_FPS)))

            bounding_boxes,_ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            #   for each box
            for (x1, y1, x2, y2, acc) in bounding_boxes:
                w = x2-x1
                h = y2-y1
                #   plot the box using cv2
                cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),
                    int(y1+h)),(255,0,0),2)
                print ('Accuracy score', acc)

        frame_count += 1
        cv2.imshow('',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
