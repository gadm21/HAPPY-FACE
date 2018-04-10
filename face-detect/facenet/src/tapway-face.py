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
import sys
import numpy as np
import aws.rekognition as aws
import base64
import json
import math
import dlib
import threading

gpu_memory_fraction = 1.0
# minsize = 50 ## default value
minsize = 40
# threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
factor = 0.709 # scale factor

def saveImage(saveImgPath,cropFace,numFace):
    if os.path.isdir(saveImgPath):
        cv2.imwrite('{}/face_{}.png'.format(saveImgPath,numFace),cropFace)      
    else:
        print('{} path does not exists.'.format(saveImgPath))

def doRecognizePerson(faceNames,fid):
    time.sleep(2)
    faceNames[fid] = "Person "+str(fid)

def AWSRekognition(enc):
    result  = aws.search_faces(enc)
    if len(result['FaceMatches']) == 0:
        print('New face add to collection')
        res = aws.index_faces(enc)
        # print(res)
    else:
        print('Match Face ID {}'.format(result['FaceMatches'][0]['Face']['FaceId']))

### create image that min 80*80 pixel for aws rekognition
def cropFaceAWS(img,bb,crop_factor):
    height,width,_ = img.shape

    x1 = bb[0]
    x2 = bb[1]
    y1 = bb[2]
    y2 = bb[3]

    w = x2-x1
    h = y2-y1

    crop_y1 = int(y1-crop_factor*h)
    crop_y2 = int(y2+crop_factor*h)
    crop_x1 = int(x1-crop_factor*w)
    crop_x2 = int(x2+crop_factor*w) 

    if crop_y1 < 0:
        crop_y1 = 0

    if crop_y2 > height:
        crop_y2 = height

    if crop_x1 < 0:
        crop_x1 = 0

    if crop_x2 > width:
        crop_x2 = width

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

    return cropface

def detect_Face():
    #file = "http://192.168.0.4/video/mjpg.cgi"
    file = 0
    saveImgPath = 'face_img'

    frame_interval = 10
    frame_count = 0

    mainWindow = 'Base Image'
    sideWindow = 'Cropped Image'

    camera = cv2.VideoCapture(file)
    cv2.namedWindow(mainWindow,0)
    cv2.namedWindow(sideWindow)
    # cv2.namedWindow('result-image',0)

    _,frame = camera.read()
    height,width,_ = frame.shape

    cv2.resizeWindow(mainWindow,width,height)
    cv2.resizeWindow(sideWindow,width,height)
    cv2.moveWindow(mainWindow,0,0)
    cv2.moveWindow(sideWindow,100,400)
    
    cv2.startWindowThread()

    num_face = 0
    currentFaceID = 0
    faceTrackers = {}
    faceNames = {}
    rectangleColor = (0,165,255)

    while True:
        _, frame = camera.read()
        ### Avoid use imgDisplay = frame
        imgDisplay = frame.copy()
        frame_count += 1

        fidsToDelete = []
        for fid in faceTrackers.keys():
            trackingQuality = faceTrackers[fid].update(imgDisplay)
            # print('Quality '+str(trackingQuality))
            if trackingQuality < 5:
                fidsToDelete.append(fid)

        for fid in fidsToDelete:
            print('Removing fid '+str(fid)+' from list of trackers')
            faceTrackers.pop(fid,None)

        if (frame_count%frame_interval) == 0:
            bounding_boxes,_ = align.detect_face.detect_face(imgDisplay, minsize, pnet, rnet, onet, threshold, factor)

            #   for each box
            for (x1, y1, x2, y2, acc) in bounding_boxes:
                try:

                    w = x2-x1
                    h = y2-y1

                    w = int(x2-x1)
                    h = int(y2-y1)
                    x = int(x1)
                    y = int(y1)

                    ##calculate centerpoint
                    x_bar  =x+0.5*w
                    y_bar = y+0.5*h

                    matchedFid = None

                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()
                        
                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        t_x_bar = t_x+0.5*t_w
                        t_y_bar = t_y+0.5*t_h

                        if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
                             ( t_y <= y_bar   <= (t_y + t_h)) and 
                             ( x   <= t_x_bar <= (x   + w  )) and 
                             ( y   <= t_y_bar <= (y   + h  ))):
                            matchedFid = fid
                    
                    if matchedFid is None:
                        print('Creating new tracker'+str(currentFaceID))
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(imgDisplay,dlib.rectangle(x-10,y-20,x+w+10,y+h+20))
                        faceTrackers[currentFaceID] = tracker

                        t = threading.Thread(target=doRecognizePerson,args=(faceNames,currentFaceID))
                        t.start()
                        currentFaceID += 1

                        crop_factor = 0.25
                        cropface = cropFaceAWS(frame,(x1,x2,y1,y2),crop_factor)
                        saveImage(saveImgPath,cropface,num_face)
                        num_face += 1
                            
                        enc = cv2.imencode('.png',cropface)[1].tostring()
                        # cc = base64.b64encode(cc)
                        t2 = threading.Thread(target=AWSRekognition,args=([enc]))
                        t2.start()
   
                    cv2.imshow(sideWindow,cropface)
     
                    # cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),
                    #     int(y1+h)),(255,0,0),2)

                except Exception as e:
                    #using except statement become index_faces from aws rekognition will have error if does not detect any face
                    print(e)
                    pass
        
        for fid in faceTrackers.keys():
            tracked_position = faceTrackers[fid].get_position()
            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            cv2.rectangle(imgDisplay, (t_x, t_y),
                                    (t_x + t_w , t_y + t_h),
                                    rectangleColor ,2)


            if fid in faceNames.keys():
                cv2.putText(imgDisplay, faceNames[fid] , 
                            (int(t_x + t_w/2), int(t_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
            else:
                cv2.putText(imgDisplay, "Detecting..." , 
                            (int(t_x + t_w/2), int(t_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

        cv2.imshow(mainWindow,imgDisplay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
            log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(
                sess, None)
        detect_Face()