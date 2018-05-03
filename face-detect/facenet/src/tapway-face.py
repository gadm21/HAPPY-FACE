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
import threading
import track.tracker as track

import gui.tkgui as tkgui
import tkinter as tk
from PIL import ImageTk,Image

gpu_memory_fraction = 1.0
minsize = 40
# threshold = [ 0.1, 0.2, 0.0001 ]  # three steps's threshold
threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
factor = 0.709 # scale factor

def saveImage(saveImgPath,cropFace,numFace):
	if os.path.isdir(saveImgPath):
		cv2.imwrite('{}/face_{}.png'.format(saveImgPath,numFace),cropFace)      
	else:
		print('{} path does not exists.'.format(saveImgPath))

def doRecognizePerson(faceNames,fid):
	time.sleep(2)
	# faceNames[fid] = "Person "+str(fid)

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

class GUI(tk.Tk):
	def __init__(self, *args, **kwargs):
		root = tk.Tk.__init__(self, *args, **kwargs)
		self.file = 0
		# self.file = "rtsp://admin:admin@192.168.0.8/play1.sdp"

		self.detect = FaceDetect()
		self.winfo_toplevel().resizable(width=False,height=False)

		self.winfo_toplevel().title('Tapway Face System')
		self.videoFrame = tk.Frame(root,width=600,height=600,borderwidth=0)
		self.videoFrame.pack(side=tk.LEFT)
		
		self.videoLabel = tk.Label(self.videoFrame)
		self.videoLabel.pack()
		self.camera = cv2.VideoCapture(self.file)
		
		self.frame = tkgui.VerticalScrolledFrame(root)
		self.frame.pack(side=tk.RIGHT)

		self.imageList = []

		### read single frame to setup imageList
		_,frame = self.camera.read()

		height,_,_ = frame.shape
		row = math.ceil(height/120)

		self.addImageList(row*2)

	def addImageList(self,row):
		photo = cv2.imread('gui/icon.jpg')

		height,width,_ = photo.shape
		scaleY = 100.0/height
		scaleX = 100.0/width
		resizeImg = cv2.resize(photo,(0,0),fx=scaleX,fy=scaleY)
		cv2image = cv2.cvtColor(resizeImg,cv2.COLOR_BGR2RGBA)

		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)

		for i in range(row):
			lb = tk.Label(self.frame.interior,image=imgtk)
			lb.imgtk = imgtk
			if i%2 == 0:
				lb.grid(row=int(i/2),column=i%2,padx=(10,5),pady=10)
			else:
				lb.grid(row=int(i/2),column=i%2,padx=(5,10),pady=10)

			self.imageList.append(lb)

	def show_frame(self):
		flag, frame = self.camera.read()
		if flag == True:
			### facenet accept img shape (?,?,3)
			self.detect.detectFace(frame,self.imageList,self.frame.interior)
			### convert img shape to (?,?,4)
			cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)

			img = Image.fromarray(cv2image)
			imgtk = ImageTk.PhotoImage(image=img)
			self.videoLabel.imgtk = imgtk
			self.videoLabel.configure(image=imgtk)
		else:
			print('No frame come in')
		self.videoLabel.after(10,self.show_frame)

class FaceDetect:
	def __init__(self,*args,**kwargs):
		self.saveImgPath = 'face_img'

		self.frame_interval = 10
		self.frame_count = 0

		self.num_face = 0
		self.currentFaceID = 0

		self.tracker = track.Tracker()
		self.crop_factor = 0.25

	def AWSRekognition(self,enc,fid):
		try:
			result  = aws.search_faces(enc)
			if len(result['FaceMatches']) == 0:
				print('New face add to collection')
				res = aws.index_faces(enc)
				print('Face ID:',res['FaceRecords'][0]['Face']['FaceId'])
				self.tracker.faceNames[fid] = "New Face ID: "+res['FaceRecords'][0]['Face']['FaceId']
			else:
				# add aws id
				self.tracker.faceNames[fid] = "Match ID: "+result['FaceMatches'][0]['Face']['FaceId']
				print('Match Face ID {}'.format(result['FaceMatches'][0]['Face']['FaceId']))

		except Exception as e:
			# using except statement become index_faces from aws rekognition will have error if does not detect any face
			if type(e).__name__ == 'InvalidParameterException':
				### delete the face, if aws cant detect it as a face
				self.tracker.appendDeleteFid(fid)
			else:
				print(e)
			pass

	def drawTrackedFace(self,imgDisplay):
		rectangleColor = (0,165,255)

		for fid in self.tracker.faceTrackers.keys():
			tracked_position = self.tracker.faceTrackers[fid].get_position()
			t_x = int(tracked_position.left())
			t_y = int(tracked_position.top())
			t_w = int(tracked_position.width())
			t_h = int(tracked_position.height())

			cv2.rectangle(imgDisplay, (t_x, t_y),
									(t_x + t_w , t_y + t_h),
									rectangleColor ,2)

			if fid in self.tracker.faceNames.keys():
				# cv2.putText(imgDisplay, self.tracker.faceNames[fid] , 
				# 			(int(t_x + t_w/2), int(t_y)), 
				# 			cv2.FONT_HERSHEY_SIMPLEX,
				# 			0.5, (255, 255, 255), 2)
				text = self.tracker.faceNames[fid]
				textsize = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)[0]
				textX = (textsize[0])/2
				# textY = (t_w+textsize[1])/2
				cv2.putText(imgDisplay, self.tracker.faceNames[fid] , 
							(int(t_x+t_w/2-textX), int(t_y)), 
							cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (255, 255, 255), 2)
			else:
				cv2.putText(imgDisplay, "Detecting..." , 
							(int(t_x + t_w/2), int(t_y)), 
							cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (255, 255, 255), 2)

	def detectFace(self,imgDisplay,imageList,interior):
		### Avoid use imgDisplay = frame
		frame = imgDisplay.copy()

		self.frame_count += 1
		self.tracker.deleteTrack(imgDisplay)

		if (self.frame_count%self.frame_interval) == 0:
			bounding_boxes,_ = align.detect_face.detect_face(imgDisplay, minsize, pnet, rnet, onet, threshold, factor)

			#   for each box
			for (x1, y1, x2, y2, acc) in bounding_boxes:

				w = int(x2-x1)
				h = int(y2-y1)
				x = int(x1)
				y = int(y1)

				matchedFid = self.tracker.getMatchId(x,y,w,h)

				if matchedFid is None:
					self.currentFaceID += 1
					self.num_face += 1

					self.tracker.createTrack(imgDisplay,x,y,w,h,self.currentFaceID)
					t = threading.Thread(target=doRecognizePerson,args=(self.tracker.faceNames,self.currentFaceID))
					t.start()

					cropface = cropFaceAWS(frame,(x1,x2,y1,y2),self.crop_factor)
					saveImage(self.saveImgPath,cropface,self.num_face)
						
					enc = cv2.imencode('.png',cropface)[1].tostring()
					t2 = threading.Thread(target=self.AWSRekognition,args=([enc,self.currentFaceID]))
					t2.start()

					### Make sure image list will have one row more than the num face
					if len(imageList) < self.num_face+2:
						photo = cv2.imread('gui/icon.jpg')

						height,width,_ = photo.shape
						scaleY = 100.0/height
						scaleX = 100.0/width
						resizeImg = cv2.resize(photo,(0,0),fx=scaleX,fy=scaleY)
						cv2image = cv2.cvtColor(resizeImg,cv2.COLOR_BGR2RGBA)

						img = Image.fromarray(cv2image)
						imgtk = ImageTk.PhotoImage(image=img)

						for i in range(2):
							r = self.num_face+int(i/2)
							c = i%2
							lb = tk.Label(interior,image=imgtk)
							lb.imgtk = imgtk
							if i%2 == 0:
								lb.grid(row=r,column=c,padx=(10,5),pady=10)
							else:
								lb.grid(row=r,column=c,padx=(5,10),pady=10)

							imageList.append(lb)

					height,width,_ = cropface.shape
					scaleY = 100.0/height
					scaleX = 100.0/width
					resizeImg = cv2.resize(cropface,(0,0),fx=scaleX,fy=scaleY)
					cv2image = cv2.cvtColor(resizeImg,cv2.COLOR_BGR2RGBA)

					img = Image.fromarray(cv2image)
					imgtk = ImageTk.PhotoImage(image=img)

					index = self.num_face-1
					imageList[index].imgtk = imgtk
					imageList[index].configure(image=imgtk)
		
		self.drawTrackedFace(imgDisplay)

if __name__ == '__main__':
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(
			per_process_gpu_memory_fraction=gpu_memory_fraction)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
			log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = align.detect_face.create_mtcnn(
				sess, None)
		app = GUI()
		app.show_frame()
		app.mainloop()
