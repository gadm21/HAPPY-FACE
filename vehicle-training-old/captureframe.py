#import dependencies
import os
import time
import cv2
from os.path import isfile, join

timestr = time.time()

#for automatic frame capture, prints success for every frame recorded
#get frames from live rtsp feed , 3 frames per second, url can be configured
#vidcap = cv2.VideoCapture("rtsp://admin:admin123@tapway1.dlinkddns.com:1234/cam/realmonitor?channel=1&subtype=0");
vidcap = cv2.VideoCapture("20181023_165959_(13)_PB_U24.mpg");

success,image = vidcap.read()
count = 0
success = True
mypath = 'TrainingData/raw-images'

while success:
    success,image = vidcap.read()
    print('read a new frame:',success)
    if count%7 == 0 :
        cv2.imwrite(os.path.join(mypath, 'frame%d.jpg' %timestr), image)

        print('success%s' %count)
    elif count%2520 ==0 : #when 3.5 minutes,force stop frame capture
        print('time is up !')
        break
    count+=1
    timestr+=1


