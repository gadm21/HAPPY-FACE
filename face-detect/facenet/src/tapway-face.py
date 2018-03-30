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
import time
import sys ### oka import system
# import numpy as np ## for to_rgb
import aws.rekognition as aws
import base64
import json
import math

#   setup facenet parameters
gpu_memory_fraction = 1.0
# minsize = 50 ## default value
minsize = 40
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

def detect_face():
    # frame_interval = 3
    frame_interval = 10
    fps_display_interval = 10
    frame_rate = 0
    frame_count = 0

    #file = "http://192.168.0.4/video/mjpg.cgi"
    file = 0
    saveImgPath = 'face_img'
    camera = cv2.VideoCapture(file)
    cv2.namedWindow('',0)
    _,img = camera.read()
    height,width,_ = img.shape
    cv2.resizeWindow('',width,height)
    
    start_time = time.time()

    num_face = 0

    while True:
        _, img = camera.read()


        if (frame_count%frame_interval) == 0:
            # print("Frame: "+str(camera.get(cv2.CAP_PROP_FPS)))

            bounding_boxes,_ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

            height,width,channel = img.shape
            #   for each box
            for (x1, y1, x2, y2, acc) in bounding_boxes:
                try:
                    print('Face {}'.format(num_face))

                    w = x2-x1
                    h = y2-y1

                    #   plot the box using cv2
                    #cropface = img[int(y1):int(y1+h),int(x1):int(x1+w)]

                    ### increase size of crop face, since aws cant detect the faces if too small
                    crop_factor = 0.25
                    crop_y1 = int(y1-crop_factor*h)
                    crop_y2 = int(y2+crop_factor*h)
                    crop_x1 = int(x1-crop_factor*w)
                    crop_x2 = int(x2+crop_factor*w) 

                    if crop_y1 < 0:
                        # print('crop y1 solve')
                        crop_y1 = 0

                    if crop_y2 > height:
                        # print('crop y2 solve')
                        crop_y2 = height

                    if crop_x1 < 0:
                        # print('crop x1 solve')
                        crop_x1 = 0

                    if crop_x2 > width:
                        # print('crop x2 solve')
                        crop_x2 = width


                    # print('{} {} {} {}'.format(str(crop_y1),str(crop_y2),str(crop_x1),str(crop_x2)))
                    cropface = img[crop_y1:crop_y2,crop_x1:crop_x2]

                    crop_h = crop_y2-crop_y1
                    crop_w = crop_x2-crop_x1

                    border_h = 0
                    border_w = 0

                    if crop_h < 80:
                        border_h = math.ceil((80-crop_h)/2)

                    if crop_w < 80:
                        border_w = math.ceil((80-crop_w)/2)

                    if crop_h < 80 or crop_w < 80:
                        BLACK = [255,255,255]
                        cropface = cv2.copyMakeBorder(cropface,border_h,border_h,border_w,border_w,cv2.BORDER_CONSTANT,value=BLACK)

                    if os.path.isdir(saveImgPath):
                        # print('save faces')
                        cv2.imwrite('{}/face_{}.png'.format(saveImgPath,num_face),cropface)      
                        num_face += 1       
                    else:
                        print('{} path does not exists.'.format(saveImgPath))
                        return       

                    # cv2.imshow("cropface",cropface)

                    cc = cv2.imencode('.png',cropface)[1].tostring()
                    # cc = base64.b64encode(cc)
                    result = aws.search_faces(cc)
                    # print(json.dumps(result,indent=4))
                    # print(result['SearchedFaceConfidence'])
                    # print(len(result['FaceMatches']))
                    if len(result['FaceMatches']) == 0:
                        print('New face add to collection')
                        res = aws.index_faces(cc)
                        # print(res)
                    else:
                        print('Match Face ID {}'.format(result['FaceMatches'][0]['Face']['FaceId']))
                        # print(result['FaceMatches'])
                    # cv2.imshow("croppedd",cropface)

                    cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),
                        int(y1+h)),(255,0,0),2)

                    # print ('Accuracy score', acc)

                except Exception as e:
                    #using except statement become index_faces from aws rekognition will have error if does not detect any face
                    print(e)
                    pass
        cv2.imshow('',img)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
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
        detect_face()

