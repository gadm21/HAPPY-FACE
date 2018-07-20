#   import facenet libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import os
import align.detect_face

import cv2
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
import base64
import json
import math
import threading
from configparser import ConfigParser

import gui.tkgui as tkgui
import tkinter as tk
import tkinter.ttk as ttk
from PIL import ImageTk,Image

import aws.rekognition as aws
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import blurdetector.blurdetect as blurDetector
import track.tracker as track

gpu_memory_fraction = 1.0
minsize = 40
threshold = [ 0.6, 0.7, 0.7 ]
factor = 0.709

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def saveImage(saveImgPath,cropFace,numFace):
	if not os.path.isdir(saveImgPath):
		os.mkdir(saveImgPath)
		print('Create new {} path'.format(saveImgPath))
	cv2.imwrite('{}/face_{}.png'.format(saveImgPath,numFace),cropFace)      

def cropFace(img,boundingBox,crop_factor=0,minHeight=-1,minWidth=-1):
	'''
	minHeight = -1 means no minimum height for cropped image
	minHeight = 80 means if cropped image with height less than 80 
				it will add padding to the make the image meet the minimum Height
	'''
	height,width,_ = img.shape

	x1,y1,x2,y2 = boundingBox

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

	if minHeight != -1 and crop_h < minHeight:
		border_h = math.ceil((80-crop_h)/2)

	if minWidth != -1 and crop_w < minWidth:
		border_w = math.ceil((80-crop_w)/2)

	if minHeight != -1 or minWidth != -1:
		BLACK = [255,255,255]
		cropface = cv2.copyMakeBorder(cropface,border_h,border_h,border_w,border_w,cv2.BORDER_CONSTANT,value=BLACK)

	return cropface

def resizeImage(sizeX,sizeY,img):
	height,width,_ = img.shape
	scaleY = sizeY/height
	scaleX = sizeX/width
	resizeImg = cv2.resize(img,(0,0),fx=scaleX,fy=scaleY)
	return resizeImg

### Note: The three method above will move to other file in later ###
class GUI(tk.Tk):
	def __init__(self, *args, **kwargs):
		root = tk.Tk.__init__(self, *args, **kwargs)

		self.readConfigFile()

		self.frame_interval = 10
		self.frame_count = 0

		self.num_face = 0
		self.currentFaceID = 0

		self.tracker = track.Tracker()
		self.crop_factor = 0.20

		self.winfo_toplevel().resizable(width=False,height=False)
		self.winfo_toplevel().title('Tapway Face System')

		self.createMenu()

		self.videoFrame = tk.ttk.Frame(root,width=600,height=600,borderwidth=0)
		self.videoFrame.pack(side=tk.LEFT)

		self.videoLabel = tk.Label(self.videoFrame)
		self.videoLabel.pack()
		self.camera = cv2.VideoCapture(self.file)
		
		self.frame = tkgui.VerticalScrolledFrame(root)
		self.frame.pack(side=tk.RIGHT)

		self.imageList = []

		### read single frame to setup imageList
		_,frame = self.camera.read()
		self.tracker.videoFrameSize = frame.shape

		self.addImageList(2)

		self.faceNamesList = {}
		self.faceAttributesList = {}
		
		self.headPoseEstimator = CnnHeadPoseEstimator(sess)
		self.headPoseEstimator.load_roll_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
		self.headPoseEstimator.load_pitch_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
		self.headPoseEstimator.load_yaw_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))

	def createMenu(self):
		menu = tk.Menu(self.winfo_toplevel())
		 
		fileMenu = tk.Menu(menu)
		fileMenu.add_command(label='New')
		 
		menu.add_cascade(label='File', menu=fileMenu)
		 
		self.winfo_toplevel().config(menu=menu)

		editMenu = tk.Menu(menu)
		editMenu.add_separator()
		editMenu.add_command(label="Edit Filter Parameters", command = lambda :self.filterParameterPopup())
		editMenu.add_command(label='Configure IP Address', command = lambda : self.changeIPPopup())

		menu.add_cascade(label="Edit", menu=editMenu)

	def changeIPPopup(self):
		popup = tk.Toplevel()
		popup.wm_title('Configure IP Address')
		label = ttk.Label(popup, text='Enter IP address: ')
		label.pack(side=tk.LEFT)
		ip = tk.StringVar(None)
		input = ttk.Entry(popup, textvariable = ip,width = 40)
		input.pack(side=tk.LEFT)
		popup.bind('<Return>',lambda _:self.updateIP(ip=ip.get()) or popup.destroy())
		button = ttk.Button(popup, text ='OK', command = lambda :self.updateIP(ip=ip.get()) or popup.destroy())
		button.pack()

	def updateIP(self, ip):
		if(isInt(ip)):
			self.file = int(ip)
		else:
			self.file = ip
		self.camera.release()
		self.camera = cv2.VideoCapture(self.file)
		print('IP: ',self.file)

	def filterParameterPopup(self):
		newPopup = tk.Toplevel()
		newPopup.wm_title("Filter Parameter")
		newPopup.geometry("300x300")
		label = ttk.Label(newPopup,text="Pitch Angle: ", font=("Helvetica",10))
		label.pack(pady=10)
		value = ttk.Label(newPopup,textvariable=self.pitchFilter)
		value.pack()
		pitchScale = ttk.Scale(newPopup, from_=0, to=90, orient=tk.HORIZONTAL,variable = self.pitchFilter, command=lambda _:self.updatePitch(pitch=pitchScale.get()))
		pitchScale.set(self.pitchFilter)
		pitchScale.pack()
		label2 = ttk.Label(newPopup, text="Yaw Angle: ", font=("Helvetica",10))
		label2.pack(pady=10)
		value2 = ttk.Label(newPopup, textvariable=self.yawFilter)
		value2.pack()
		yawScale = ttk.Scale(newPopup, from_=0, to=90, orient=tk.HORIZONTAL,variable = self.yawFilter, command=lambda _:self.updateYaw(yaw=yawScale.get()))
		yawScale.set(self.yawFilter)
		yawScale.pack()
		label3 = ttk.Label(newPopup, text="Blurriness Factor: ", font=("Helvetica",10))
		label3.pack(pady=10)
		value3 = ttk.Label(newPopup, textvariable=self.blurFilter)
		value3.pack()
		blurScale = ttk.Scale(newPopup, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.blurFilter, command=lambda _:self.updateBlur(blur=blurScale.get()))
		blurScale.set(self.blurFilter)
		blurScale.pack()

		newPopup.mainloop()

	def updatePitch(self, pitch):
		self.pitchFilter=pitch
		self.printFilter()

	def updateYaw(self, yaw):
		self.yawFilter=yaw
		self.printFilter()

	def updateBlur(self, blur):
		self.blurFilter=blur
		self.printFilter()

	def printFilter(self):
		print("Current pitch filter: {}".format(self.pitchFilter))
		print("Current yaw filter: {}".format(self.yawFilter))
		print("Current blur filter: {}\n".format(self.blurFilter))

	def readConfigFile(self):
		config = ConfigParser()
		config.read('config.ini')

		self.file = config.get('default','video')

		if self.file.isdigit():
			self.file = int(self.file)

		self.saveImgPath = config.get('default','imagePath')
		self.pitchFilter = config.getfloat('default','pitchFilter')
		self.yawFilter = config.getfloat('default','yawFilter') 
		self.blurFilter = config.getfloat('default','blurFilter')

	def addImageList(self,row):
		photo = cv2.imread('gui/icon.jpg')
		cv2image = resizeImage(100,100,photo)

		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)

		for i in range(row):
			imgLabel = tk.ttk.Button(self.frame.interior,image=imgtk,state='disable')
			imgLabel.imgtk = imgtk
			self.imageList.append(imgLabel)
		self.gridResetLayout()

	def addFaceToImageList(self,fid,cropface):
		cv2image = resizeImage(100,100,cropface)
		cv2image = cv2.cvtColor(cv2image,cv2.COLOR_BGR2RGBA)

		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)
		imgLabel = tk.ttk.Button(self.frame.interior,image=imgtk,command=lambda:self.popup(fid,imgtk))
		imgLabel.imgtk = imgtk

		self.imageList.append(imgLabel)
		self.gridResetLayout()

	def gridResetLayout(self):
		### reverse image list for make the latest image in the top of list for GUI view
		for index,item in enumerate(reversed(self.imageList)):
			item.grid(row=0,column=0)
			if index%2==0:
				item.grid(row=int(index/2),column=index%2,padx=(10,5),pady=10)
			else:
				item.grid(row=int(index/2),column=index%2,padx=(5,10),pady=10)

	def showFrame(self):
		flag, frame = self.camera.read()
		if flag == True:
			### facenet accept img shape (?,?,3)
			self.detectFace(frame)
			### convert img shape to (?,?,4)
			cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)

			img = Image.fromarray(cv2image)
			imgtk = ImageTk.PhotoImage(image=img)
			self.videoLabel.imgtk = imgtk
			self.videoLabel.configure(image=imgtk)
		else:
			print('No frame come in')
		self.videoLabel.after(10,self.showFrame)

	def AWSRekognition(self,enc,cropface,fid):
		try:
			res  = aws.search_faces(enc)
			if len(res['FaceMatches']) == 0:
				res = aws.index_faces(enc)
				awsID = res['FaceRecords'][0]['Face']['FaceId']

				self.tracker.faceID[fid] = awsID
				self.faceAttributesList[fid].awsID = awsID
				self.addFaceToImageList(fid,cropface)

				print('New Face ID:',awsID)
			else:
				awsID = res['FaceMatches'][0]['Face']['FaceId']

				self.tracker.faceID[fid] = awsID
				self.faceAttributesList[fid].awsID = awsID
				self.faceAttributesList[fid].similarity = res['FaceMatches'][0]['Similarity']
				self.addFaceToImageList(fid,cropface)

				print('Match Face ID {}'.format(awsID))

		except Exception as e:
			# aws rekognition will have error if does not detect any face
			if type(e).__name__ == 'InvalidParameterException':
				print('aws exception')
				# cv2.imshow('name',cropface)
				### delete the face, since aws cant detect it as a face
				self.tracker.appendDeleteFid(fid)
				self.faceAttributesList.pop(fid,None)
			else:
				print(e)
			pass

	def drawTrackedFace(self,imgDisplay,points):

		for fid in self.tracker.faceTrackers.keys():
			tracked_position = self.tracker.faceTrackers[fid].get_position()
			t_x = int(tracked_position.left())
			t_y = int(tracked_position.top())
			t_w = int(tracked_position.width())
			t_h = int(tracked_position.height())

			text = '...Detecting...'
			rectColor = (0,165,255)

			if fid in self.tracker.faceID.keys():

				awsID = self.tracker.faceID[fid]

				if awsID in self.faceNamesList.keys():
					text = self.faceNamesList[awsID]
					rectColor = (255,0,0)
				else:
					text = awsID
					rectColor = (0,0,255)

			textSize = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)[0]
			textX = int(t_x+t_w/2-(textSize[0])/2)
			textY = int(t_y)
			textLoc = (textX,textY)

			cv2.rectangle(imgDisplay, (t_x, t_y),
									(t_x + t_w , t_y + t_h),
									rectColor ,2)

			cv2.putText(imgDisplay, text, textLoc, 
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (255, 255, 255), 2)

		a = points.shape
		if a[0] > 0:
			# print(points.shape)
			if a[1] > 0:

				for i in range(5):
					# print(points.shape)
					cv2.circle(imgDisplay, (points[i,0],points[i+5,0]), 1, (255, 0, 0), 2)

	def popup(self,fid,imgtk):

		faceAttr = self.faceAttributesList[fid]

		win = tk.Toplevel()
		win.wm_title("Register Face")

		frame = tk.ttk.Frame(win,border=2,relief=tk.GROOVE)
		frame.pack(fill=tk.X,padx=5,pady=5)

		imgLabel = tk.Label(frame,image=imgtk)
		imgLabel.imgtk = imgtk
		imgLabel.pack(fill=tk.X,pady=5)

		nameFrame = tk.ttk.Frame(frame)
		nameFrame.pack(fill=tk.X,pady=5)

		nameLabel = tk.ttk.Label(nameFrame,text='{0:15}\t:'.format('Name'))
		nameLabel.pack(side=tk.LEFT,padx=5)

		nameEntry = tk.ttk.Entry(nameFrame)
		nameEntry.pack(side=tk.LEFT)

		faceid = tk.ttk.Label(frame,text='{0:15}\t: {1}'.format('Face ID',faceAttr.awsID))
		faceid.pack(fill=tk.X,padx=5)

		similarityFrame = tk.ttk.Label(frame,text='{0:15}\t: {1}'.format('Similarity',faceAttr.similarity))
		similarityFrame.pack(fill=tk.X,padx=5)

		rollFrame = tk.ttk.Label(frame,text='{0:15}\t: {1}'.format('Roll(degree)',faceAttr.roll))
		rollFrame.pack(fill=tk.X,padx=5)

		yawFrame = tk.ttk.Label(frame,text='{0:15}\t: {1}'.format('Yaw(degree)',faceAttr.yaw))
		yawFrame.pack(fill=tk.X,padx=5)

		pitchFrame = tk.ttk.Label(frame,text='{0:15}\t: {1}'.format('Pitch(degree)',faceAttr.pitch))
		pitchFrame.pack(fill=tk.X,padx=5)

		submit = tk.ttk.Button(frame,text='Submit',width=10,command=lambda:self.submit(faceAttr.awsID,nameEntry.get(),win))
		submit.pack(pady=5)

	def submit(self,awsID,name,win):
		self.faceNamesList[awsID] = name
		win.destroy();

	def getHeadPoseEstimation(self,faceImg):
		'''
		Image requirement for Head Pose Estimator
		[github link: https://github.com/mpatacchiola/deepgaze ]
		1) Same width and height
		2) Three color channel
		3) Minimum size 64x64
		''' 
		resizeFaceImg = resizeImage(64,64,faceImg)

		roll = self.headPoseEstimator.return_roll(resizeFaceImg)
		pitch = self.headPoseEstimator.return_pitch(resizeFaceImg)
		yaw = self.headPoseEstimator.return_yaw(resizeFaceImg)

		return roll[0,0,0], pitch[0,0,0], yaw[0,0,0]

	def detectBlur(self,img,minZero=0.015,thresh=35):
		'''
		Code Implementation from Paper
		[link: https://www.cs.cmu.edu/~htong/pdf/ICME04_tong.pdf]
		'''
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		Emax1, Emax2, Emax3 = blurDetector.algorithm(gray)
		per, BlurExtent = blurDetector.ruleset(Emax1, Emax2, Emax3, thresh)

		print('BlurExtent: {}\tper: {}'.format(BlurExtent,per))
		
		if per > minZero:
			return False
		else:
			return True

	def detectFace(self,imgDisplay):
		### Avoid use imgDisplay = frame
		frame = imgDisplay.copy()

		self.frame_count += 1
		self.tracker.deleteTrack(imgDisplay)

		points = np.empty((0,0))

		if (self.frame_count%self.frame_interval) == 0:
			bounding_boxes,points = align.detect_face.detect_face(imgDisplay, minsize, pnet, rnet, onet, threshold, factor)
			
			for (x1, y1, x2, y2, acc) in bounding_boxes:

				matchedFid = self.tracker.getMatchId(imgDisplay,(x1,y1,x2,y2))

				if matchedFid is None:
					faceAttr = FaceAttribute()

					self.currentFaceID += 1
					self.num_face += 1

					faceImg = cropFace(frame,(x1,y1,x2,y2))

					blur = self.detectBlur(faceImg,minZero=0.0,thresh=self.blurFilter)

					if blur:
						print('Filter Blur Image')
						saveImage('blur_img',faceImg,self.num_face)
						continue
					
					# print('Sharp')
					saveImage('sharp_img',faceImg,self.num_face)

					roll,pitch,yaw = self.getHeadPoseEstimation(faceImg)

					if abs(yaw) > self.yawFilter or abs(pitch) > self.pitchFilter:
						print('Exceed Yaw | Pitch Threshold!')
						continue

					faceAttr.roll = float(roll)
					faceAttr.pitch = float(pitch)
					faceAttr.yaw = float(yaw)

					self.faceAttributesList[self.currentFaceID] = faceAttr

					self.tracker.createTrack(imgDisplay,(x1,y1,x2,y2),self.currentFaceID)

					cropface = cropFace(frame,(x1,y1,x2,y2),crop_factor=self.crop_factor,minHeight=80,minWidth=80)

					saveImage(self.saveImgPath,cropface,self.num_face)

					enc = cv2.imencode('.png',cropface)[1].tostring()

					t2 = threading.Thread(target=self.AWSRekognition,args=([enc,cropface,self.currentFaceID]))
					t2.start()

		self.drawTrackedFace(imgDisplay,points.copy())

class FaceAttribute(object):
	def __init__(self):
		self.awsID = None
		self.yaw = None
		self.roll = None
		self.pitch = None
		#self.blurExtent = None # not yet update
		self.similarity = 'New Face'

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
		app.showFrame()
		app.mainloop()
