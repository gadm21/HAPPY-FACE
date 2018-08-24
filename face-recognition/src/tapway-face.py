#   import facenet libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import pickle

import tensorflow as tf
import os
import traceback
import align.detect_face

import cv2
import time
import sys
import numpy as np
import json
import jsonpickle
import math
import logging.config
import threading
from configparser import ConfigParser

import gui.tkgui as tkgui
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from PIL import ImageTk,Image

import aws.rekognition as aws
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import blurdetector.blurdetect as blurDetector
import track.tracker as track

gpu_memory_fraction = 1.0
minsize = 40
threshold = [ 0.6, 0.7, 0.7 ]
factor = 0.709

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
age_list=[[0, 2],[4, 6],[8, 12],[15, 20],[25, 32],[38, 43],[48, 53],[60, 100]]

gender_list = ['Male', 'Female']

logging.config.fileConfig('logging.conf')
logging.logThreads = 0
logging.logProcesses= 0
logging._srcfile = None
logger = logging.getLogger('tapway-face')

def exc_handler(type,value,tb):
	tbList = traceback.format_tb(tb)
	tbString = ''.join(tbList)
	logger.critical(
		'Caught exception {} within program: {} \nFull Traceback:\n{}\n'.format(str(value.__class__.__name__), str(value), tbString))

sys.excepthook = exc_handler

def isInt(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

def saveImage(saveImgPath,cropFace,numFace):
	if not os.path.isdir(saveImgPath):
		os.mkdir(saveImgPath)
		logger.warning('Path to {} is created as it does not exist'.format(saveImgPath))
	cv2.imwrite('{}/face_{}.png'.format(saveImgPath,numFace),cropFace)
	logger.info('Image of face number {} is saved to {}'.format(numFace,saveImgPath))

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
		logger.info('Initializing Tkinter GUI and loading all libraries')
		root = tk.Tk.__init__(self, *args, **kwargs)
		self.protocol("WM_DELETE_WINDOW", self._quit)
		self.appIcon = tk.Image("photo", file="gui/tapway.png")
		# self.call('wm','iconphoto',self._w, self.appIcon)
		self.wm_iconphoto(True,self.appIcon)

		self.readConfigFile()

		self.frame_count = 0

		self.tracker = track.Tracker()
		self.crop_factor = 0.20

		self.winfo_toplevel().title('Tapway Face System')

		self.featureOption = tk.IntVar(value=0)

		self.createMenu()
		logger.info('Reading a single frame to define frame size')
		### read single frame to setup imageList
		self.camera = cv2.VideoCapture(self.file)
		self.camera.set(cv2.CAP_PROP_BUFFERSIZE,2)
		_,frame = self.camera.read()
		self.tracker.videoFrameSize = frame.shape
		height,width,_ = frame.shape
		self.ROIx1=0
		self.ROIy1=0
		self.ROIx2=width
		self.ROIy2=height
		logger.info('Video input has resolution {} x {}'.format(width,height))

		self.outerFrame = tk.Frame(root)
		self.outerFrame.pack(side=tk.RIGHT,fill='y',expand=False)
		self.frame = tkgui.VerticalScrolledFrame(self.outerFrame)
		self.frame.pack(fill='both',expand=False)

		self.videoLabel = tk.Canvas(root, width=frame.shape[1], height=frame.shape[0],highlightthickness=0)
		self.videoLabel.pack(side=tk.RIGHT,fill='both',expand=True)
		self.videoLabel.image = None
		self.videoError = False

		self.imageList = []
		self.addImageList(2)
		self.loadDataFromFile()

		logger.info('Loading head pose estimator CNN models')
		self.headPoseEstimator = CnnHeadPoseEstimator(sess)
		self.headPoseEstimator.load_roll_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
		self.headPoseEstimator.load_pitch_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
		self.headPoseEstimator.load_yaw_variables(os.path.realpath("deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))

		logger.info('Loading age and gender estimation caffe models')
		self.ageNet = cv2.dnn.readNetFromCaffe(
			"models/age/deploy.prototxt",
			"models/age/age_net.caffemodel")

		self.genderNet = cv2.dnn.readNetFromCaffe(
			"models/gender/deploy.prototxt",
			"models/gender/gender_net.caffemodel")

		logger.info('Initialization and loading completed')

	def report_callback_exception(self, exc, val, tb):
		if str(val) != "can't invoke \"winfo\" command: application has been destroyed":
			tbList = traceback.format_tb(tb)
			tbString = ''.join(tbList)
			logger.critical(
				'Caught exception {} within tkinter thread: {} \nFull Traceback:\n{}\n'.format(str(val.__class__.__name__),str(val),tbString))

	def _quit(self):
		try:
			logger.info('Saving data before destroying window')
			self.saveDataToFile()
			self.destroy()
			logger.info('Destroyed window')
		except:
			logger.error('Unknown error occured when quitting window: '+str(sys.exc_info()[0]))

	def saveDataToFile(self):
		logger.info('Saving face attributes data to /data/faceAttr.json')
		with open('data/faceAttr.json', 'w+') as outfile:
			json.dump(jsonpickle.encode(self.faceAttributesList), outfile)

		logger.info('Saving face names data to /data/faceName.json')
		with open('data/faceName.json','w+') as outfile:
			json.dump(jsonpickle.encode(self.faceNamesList),outfile)

		logger.info('Saving image cards to /data/imageCard.pickle')
		with open('data/imageCard.pickle', 'wb+') as handle:
			pickle.dump(self.savingImageData, handle, protocol=pickle.HIGHEST_PROTOCOL)

		logger.info('Saved data to json and pickle files in /data')

	def loadDataFromFile(self):
		if not os.path.isdir('data'):
			os.makedirs('data')
		try:
			logger.info('Loading face attributes data from /data/faceAttr.json')
			with open('data/faceAttr.json') as infile:
				self.faceAttributesList = jsonpickle.decode(json.load(infile))

			logger.info('Loading face names data from /data/faceName.json')
			with open('data/faceName.json') as infile:
				self.faceNamesList = jsonpickle.decode(json.load(infile))

			logger.info('Loading image cards from /data/imageCard.pickle')
			with open('data/imageCard.pickle', 'rb') as handle:
				self.savingImageData = pickle.load(handle)
			sortedImg = [(k, self.savingImageData[k]) for k in sorted(self.savingImageData, key=self.savingImageData.__getitem__)]
			for key, card in sortedImg:
				self.addFaceToImageList(card.fid, card.image)

			logger.info('Loaded json and pickle data from files in /data')
		except Exception as ex:
			logger.warning('Failed to load data from files: '+str(ex))
			self.faceNamesList = {}
			self.faceAttributesList = {}
			self.savingImageData = {}
		max = 0
		for keys in self.faceAttributesList.keys():
			if int(keys)>max:
				max = int(keys)
		self.currentFaceID=max
		self.num_face = max

	def createMenu(self):
		menu = tk.Menu(self.winfo_toplevel())

		fileMenu = tk.Menu(menu)
		fileMenu.add_command(label='New')

		menu.add_cascade(label='File', menu=fileMenu)

		self.winfo_toplevel().config(menu=menu)

		editMenu = tk.Menu(menu)
		editMenu.add_separator()
		editMenu.add_command(label='Configure IP Address', command = lambda : self.changeIPPopup())
		editMenu.add_command(label='Feature Option',command=lambda:self.featureOptionPopup())
		editMenu.add_command(label="Edit Filter Parameters", command = lambda :self.filterParameterPopup())
		editMenu.add_command(label='Delete AWS Recognition Data', command = lambda : self.deleteAWSRecognitionPopup())
		editMenu.add_command(label='Clear Local Recognition Data', command = lambda : self.deleteLocalData())
		editMenu.add_command(label='Select Region of Interest',command = lambda : self.selectROIPopup())

		menu.add_cascade(label="Edit", menu=editMenu)

	def selectROIPopup(self):
		flag,frame=self.camera.read()
		if flag ==True:
			self.videoLabel.update()
			wwidth = self.videoLabel.winfo_width()
			wheight =  self.videoLabel.winfo_height()
			cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
			img = Image.fromarray(cv2image)
			popup = tk.Toplevel()
			popup.wm_title('Select Region of Interest')
			# popup.geometry('{}x{}'.format(wwidth,wheight))
			popup.resizable(width=False, height=False)
			drawer = tkgui.ROIDrawer(popup,img,wwidth,wheight)
			drawer.pack(side= 'top',fill='both',expand=True)
			drawer.canvas.bind("<ButtonRelease-1>", lambda _: self.handleReleaseEvent(drawer, popup))
		else:
			message = tk.messagebox.showwarning('Invalid Video Input', "Please specify correct input in 'Edit' > 'Configure IP Address' before defining ROI.")

	def handleReleaseEvent(self, drawer, oldpopup):
		x1,y1,x2,y2 = drawer.getRectangleCoor()
		oldpopup.destroy()
		popup = tk.Toplevel()
		popup.resizable(width=False, height=False)
		popup.wm_title('ROI Coordinates Confirmation')

		def validate(val):
			if str.isdigit(val):
				return True
			else:
				return False

		vcmd = popup.register(validate)

		x1frame = ttk.Frame(popup)
		x1frame.pack(fill = tk.X, pady = 5,padx=5)
		x1label = ttk.Label(x1frame, text='X1 Coordinate: ')
		x1label.pack(side=tk.LEFT, padx =5)
		x1entry = ttk.Entry(x1frame,validate='key',validatecommand=(vcmd, '%P'))
		x1entry.insert(0,str(x1))
		x1entry.selection_range(0, tk.END)
		x1entry.pack(side=tk.LEFT)

		y1frame = ttk.Frame(popup)
		y1frame.pack(fill=tk.X,pady=5,padx=5)
		y1label = ttk.Label(y1frame, text='Y1 Coordinate: ')
		y1label.pack(side=tk.LEFT, padx=5)
		y1entry = ttk.Entry(y1frame,validate='key', validatecommand=(vcmd, '%P'))
		y1entry.insert(0, str(y1))
		y1entry.pack(side=tk.LEFT)

		x2frame = ttk.Frame(popup)
		x2frame.pack(fill=tk.X, pady=5,padx=5)
		x2label = ttk.Label(x2frame, text='X2 Coordinate: ')
		x2label.pack(side=tk.LEFT, padx=5)
		x2entry = ttk.Entry(x2frame, validate='key', validatecommand=(vcmd, '%P'))
		x2entry.insert(0, str(x2))
		x2entry.pack(side=tk.LEFT)

		y2frame = ttk.Frame(popup)
		y2frame.pack(fill=tk.X, pady=5,padx=5)
		y2label = ttk.Label(y2frame, text='Y2 Coordinate: ')
		y2label.pack(side=tk.LEFT, padx=5)
		y2entry = ttk.Entry(y2frame, validate='key', validatecommand=(vcmd, '%P'))
		y2entry.insert(0, str(y2))
		y2entry.pack(side=tk.LEFT)

		popup.bind('<Return>', lambda _: self.updateROI(x1entry.get(),y1entry.get(),x2entry.get(),y2entry.get()) or popup.destroy())
		button = ttk.Button(popup, text='OK', command=lambda: self.updateROI(x1entry.get(),y1entry.get(),x2entry.get(),y2entry.get()) or popup.destroy())
		button.pack(pady = 5)

	def updateROI(self,x1,y1,x2,y2):
		height, width, _ = self.tracker.videoFrameSize
		x1 = int(x1)
		y1 = int(y1)
		x2 = int(x2)
		y2 = int(y2)
		if x1 < 0:
			x1=0
		if y1<0:
			y1=0
		if x2>width:
			x2=width
		if y2>height:
			y2=height
		self.ROIx1 = x1
		self.ROIy1 = y1
		self.ROIx2 = x2
		self.ROIy2 = y2

		logger.info('ROI has been set to ({},{}) ({},{})'.format(x1,y1,x2,y2))

	def changeIPPopup(self):
		popup = tk.Toplevel()
		popup.resizable(width=False,height=False)
		popup.wm_title('Configure IP Address')
		label = ttk.Label(popup, text='Enter IP address: ')
		label.pack(side=tk.LEFT)
		ip = tk.StringVar(None)
		ip.set(str(self.file))
		input = ttk.Entry(popup, textvariable = ip,width = 40)
		input.selection_range(0,tk.END)
		input.pack(side=tk.LEFT)
		popup.bind('<Return>',lambda _:self.updateIP(ip=ip.get()) or popup.destroy())
		button = ttk.Button(popup, text ='OK', command = lambda :self.updateIP(ip=ip.get()) or popup.destroy())
		button.pack()
		input.focus()

	def updateIP(self, ip):
		if ip=='':
			pass
		elif(isInt(ip)):
			self.file = int(ip)
		else:
			self.file = ip
		self.camera.release()
		self.camera = cv2.VideoCapture(self.file)
		self.camera.set(cv2.CAP_PROP_BUFFERSIZE,2)
		flag, frame = self.camera.read()
		self.videoError = False
		if flag:
			height, width, _ = frame.shape
			if (self.tracker.videoFrameSize != frame.shape):
				self.tracker.videoFrameSize = frame.shape
				self.videoLabel.config(width=frame.shape[1],height=frame.shape[0])
				self.ROIx1 = 0
				self.ROIy1 = 0
				self.ROIx2 = width
				self.ROIy2 = height
			logger.info('New IP has been set to {} by user and has resolution {} x {}'.format(self.file,width,height))
		else:
			logger.error('New IP has been set to {} by user and does not has proper input'.format(self.file))

	def featureOptionPopup(self):
		popup = tk.Toplevel()
		popup.resizable(width=False,height=False)
		popup.wm_title('Feature Option')

		recognition = tk.Radiobutton(popup,text='Recognition',variable=self.featureOption,value=0,height=5,width=30,command= lambda:logger.info('Recognition feature is selected'))
		demographic = tk.Radiobutton(popup,text='Demographic',variable=self.featureOption,value=1,height=5,width=30,command= lambda:logger.info('Demographic feature is selected'))
		demoNonCloud = tk.Radiobutton(popup,text='Demographic (Non Cloud)',variable=self.featureOption,value=2,height=5,width=30,command= lambda:logger.info('Demographic feature (Non Cloud) is selected'))

		recognition.pack()
		demographic.pack()
		demoNonCloud.pack()

		popup.focus()

	def filterParameterPopup(self):
		newPopup = tk.Toplevel()
		newPopup.resizable(width=False,height=False)
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
		label3 = ttk.Label(newPopup, text="Blur Min Zero: ", font=("Helvetica",10))
		label3.pack(pady=10)
		value3 = ttk.Label(newPopup, textvariable=self.blurFilter)
		value3.pack()
		blurScale = ttk.Scale(newPopup, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.blurFilter, command=lambda _:self.updateBlur(blur=blurScale.get()))
		blurScale.set(self.blurFilter)
		blurScale.pack()
		newPopup.focus()

	def updatePitch(self, pitch):
		self.pitchFilter=pitch
		logger.info('New pitch filter value is set to {} by user'.format(pitch))

	def updateYaw(self, yaw):
		self.yawFilter=yaw
		logger.info('New yaw filter value is set to {} by user'.format(yaw))

	def updateBlur(self, blur):
		self.blurFilter=blur
		logger.info('New blur min zero filter is set to {} by user'.format(blur))

	def deleteAWSRecognitionPopup(self):
		message = tk.messagebox.askokcancel("Delete AWS Recognition Data","Are you sure you want to delete All Recognition Data on AWS Server?")
		if message:
			logger.warning('User requested to delete all recognition data on AWS Server')
			self.deleteRecognition()

	def deleteLocalData(self):
		message = tk.messagebox.askokcancel("Clear All Local Recognition Data",
											"Are you sure you want to delete All Recognition Data in local database?")
		if message:
			self.num_face = 0
			self.currentFaceID = 0
			self.faceNamesList = {}
			self.faceAttributesList = {}
			self.imageList = []
			self.savingImageData = {}
			self.tracker.deleteAll()
			self.frame.destroy()
			self.frame = tkgui.VerticalScrolledFrame(self.outerFrame)
			self.frame.pack(fill='both', expand=False)
			self.addImageList(2)
			self.saveDataToFile()
			logger.warning('User requested to delete all recognition data in database')

	def deleteRecognition(self):
		res = aws.list_faces()
		faceList = []

		for face in res['Faces']:
			faceList.append(face['FaceId'])

		if len(faceList)>0:
			aws.delete_faces(faceList)
			message = tk.messagebox.showinfo('Info','Delete Successfully')
			logger.info('All recognition data on AWS is deleted successfully')
		else:
			message = tk.messagebox.showinfo('Info', 'No AWS recognition data to be deleted')
			logger.info('No recognition data on AWS to be deleted')

	def readConfigFile(self):
		config = ConfigParser()
		config.read('config.ini')
		logger.info('Reading configuration from /config.ini')

		self.file = config.get('default','video')

		if self.file.isdigit():
			self.file = int(self.file)

		self.saveImgPath = config.get('default','imagePath')
		self.frame_interval = config.getint('default','frameInterval')
		self.pitchFilter = config.getfloat('default','pitchFilter')
		self.yawFilter = config.getfloat('default','yawFilter')
		self.blurFilter = config.getfloat('default','blurFilter')
		self.detectionThread = config.getboolean('default','detectionThread')

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
		logger.info('Image list with {} row has been created'.format(row))

	def addFaceToImageList(self,fid,cropface):
		cv2image = resizeImage(100,100,cropface)
		cv2image = cv2.cvtColor(cv2image,cv2.COLOR_BGR2RGBA)

		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)
		imgLabel = tk.ttk.Button(self.frame.interior,image=imgtk,command=lambda:self.popup(fid,imgtk))
		if self.faceAttributesList[fid].similarity == 'New Face':
			ttk.Style().configure("RB.TButton", foreground='black', background='red')
			imgLabel.configure(style="RB.TButton")
		imgLabel.imgtk = imgtk

		self.imageList.append(imgLabel)
		self.gridResetLayout()
		logger.info('New face with ID {} has been added to image list'.format(fid))

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
			self.videoError = False
			roi = frame[self.ROIy1:self.ROIy2, self.ROIx1:self.ROIx2]
			### facenet accept img shape (?,?,3)
			self.detectFace(frame, roi)
			### drawing out ROI
			cv2.rectangle(frame,(self.ROIx1,self.ROIy1),(self.ROIx2,self.ROIy2),(0,255,0),2)
			### convert img shape to (?,?,4)
			cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)

			img = Image.fromarray(cv2image)
			origin = (0, 0)
			self.videoLabel.update()
			size = (self.videoLabel.winfo_width(), self.videoLabel.winfo_height())
			### resize img if there is empty spaces
			check = False
			if self.bbox('bg') != origin + size:
				check = True
				wpercent = (size[0] / float(img.size[0]))
				hpercent = (size[1] / float(img.size[1]))
				### scale while maintaining aspect ratio
				if wpercent < hpercent:
					hsize = int((float(img.size[1]) * float(wpercent)))
					img = img.resize((max(1,size[0]), max(1,hsize)), Image.ANTIALIAS)
				else:
					wsize = int((float(img.size[0]) * float(hpercent)))
					img = img.resize((max(1,wsize), max(1,size[1])), Image.ANTIALIAS)
			self.videoLabel.image = img
			self.videoLabel.imgtk = ImageTk.PhotoImage(image=self.videoLabel.image)
			self.videoLabel.delete('bg')
			self.videoLabel.create_image(*origin, anchor='nw', image=self.videoLabel.imgtk, tags='bg')
			# self.videoLabel.tag_lower('bg', 'all')
			if check:
				self.frame.update()
				self.geometry('{}x{}'.format(int(img.width+self.frame.winfo_width()),int(img.height)))
		else:
			if not self.videoError:
				logger.error('No frame came in from video feed')
				self.videoLabel.delete('bg')
				self.videoLabel.update()
				size = (self.videoLabel.winfo_width(), self.videoLabel.winfo_height())
				self.videoLabel.create_text(size[0]/2,size[1]/2, fill='red', text="Invalid video input.\nPlease specify correct input in 'Edit' > 'Configure IP Address'.",font=("Helvetica",15),tags='bg')
			self.videoError = True
		self.videoLabel.after(10,self.showFrame)

	def AWSDetection(self,enc,cropface,fid):
		res  = aws.detect_faces(enc)
		if len(res['FaceDetails']) > 0:

			faceDetail = res['FaceDetails'][0]

			self.tracker.faceID[fid] = str(fid)
			self.faceAttributesList[fid].awsID = str(fid)

			self.faceAttributesList[fid].gender = faceDetail['Gender']['Value']
			self.faceAttributesList[fid].genderConfidence = faceDetail['Gender']['Confidence']

			self.faceAttributesList[fid].ageRangeLow = faceDetail['AgeRange']['Low']
			self.faceAttributesList[fid].ageRangeHigh = faceDetail['AgeRange']['High']
			self.faceAttributesList[fid].recognizedTime = str(datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y'))

			self.addFaceToImageList(fid,cropface)
			logger.info('Estimated age and gender of face ID {} using AWS Detection'.format(fid))
			imgCard = ImageCard()
			imgCard.fid = fid
			imgCard.image = cropface
			self.savingImageData[self.num_face] = imgCard

	def AWSRekognition(self,enc,cropface,fid):
		try:
			res  = aws.search_faces(enc)
			if len(res['FaceMatches']) == 0:
				res = aws.index_faces(enc)
				awsID = res['FaceRecords'][0]['Face']['FaceId']
				faceDetail = res['FaceRecords'][0]['FaceDetail']

				self.tracker.faceID[fid] = awsID
				self.faceAttributesList[fid].awsID = awsID
				self.faceAttributesList[fid].recognizedTime = str(datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y'))

				# self.faceAttributesList[fid].gender = faceDetail['Gender']['Value']
				# self.faceAttributesList[fid].genderConfidence = faceDetail['Gender']['Confidence']

				# self.faceAttributesList[fid].ageRangeLow = faceDetail['AgeRange']['Low']
				# self.faceAttributesList[fid].ageRangeHigh = faceDetail['AgeRange']['High']

				self.addFaceToImageList(fid,cropface)
				logger.info('New Face ID {} from AWS'.format(awsID))
			else:
				awsID = res['FaceMatches'][0]['Face']['FaceId']

				# faceAnalysis = aws.detect_faces(enc)

				self.tracker.faceID[fid] = awsID
				self.faceAttributesList[fid].awsID = awsID
				self.faceAttributesList[fid].similarity = res['FaceMatches'][0]['Similarity']
				self.faceAttributesList[fid].recognizedTime = str(datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y'))

				# self.faceAttributesList[fid].gender = faceAnalysis['FaceDetails'][0]['Gender']['Value']
				# self.faceAttributesList[fid].genderConfidence = faceAnalysis['FaceDetails'][0]['Gender']['Confidence']

				# self.faceAttributesList[fid].ageRangeLow = faceAnalysis['FaceDetails'][0]['AgeRange']['Low']
				# self.faceAttributesList[fid].ageRangeHigh = faceAnalysis['FaceDetails'][0]['AgeRange']['High']

				self.addFaceToImageList(fid,cropface)
				logger.info('Face ID {} matched'.format(awsID))

			imgCard = ImageCard()
			imgCard.fid = fid
			imgCard.image = cropface
			self.savingImageData[self.num_face] = imgCard

		except Exception as e:
			# aws rekognition will have error if does not detect any face
			if type(e).__name__ == 'InvalidParameterException':
				print('aws exception')
				logger.warning('AWS Exception, no face detected')
				# cv2.imshow('name',cropface)
				### delete the face, since aws cant detect it as a face
				self.tracker.appendDeleteFid(fid)
				self.faceAttributesList.pop(fid,None)
			else:
				logger.error(e)
			pass

	def ageGenderEstimation(self,cropface,fid):
		t1 = time.time()
		blob = cv2.dnn.blobFromImage(cropface, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

		self.ageNet.setInput(blob)
		age_preds = self.ageNet.forward()
		age = age_list[age_preds[0].argmax()]

		self.genderNet.setInput(blob)
		gender_preds = self.genderNet.forward()
		gender = gender_list[gender_preds[0].argmax()]

		print(time.time()-t1,'age gender time elapsed')

		self.tracker.faceID[fid] = str(fid)
		self.faceAttributesList[fid].awsID = str(fid)

		self.faceAttributesList[fid].gender = gender
		self.faceAttributesList[fid].genderConfidence = float(max(gender_preds[0])*100.0)

		self.faceAttributesList[fid].ageRangeLow = age[0]
		self.faceAttributesList[fid].ageRangeHigh = age[1]
		self.faceAttributesList[fid].recognizedTime = str(datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y'))

		self.addFaceToImageList(fid,cropface)
		imgCard = ImageCard()
		imgCard.fid = fid
		imgCard.image = cropface
		self.savingImageData[self.num_face] = imgCard
		logger.info('Estimated age and gender of face ID {} using non-cloud solution'.format(fid))

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

				if self.featureOption.get()!=0:
					faceAttr = self.faceAttributesList[fid]
					gender = 'Gender: {}'.format(str(faceAttr.gender))
					genderTSize = cv2.getTextSize(gender, cv2.FONT_HERSHEY_SIMPLEX,0.5,2)[0]
					genderLoc = (int(t_x + t_w / 2 - (genderTSize[0]) / 2),int(t_y-5))
					age = 'Age: {} - {}'.format(faceAttr.ageRangeLow,faceAttr.ageRangeHigh)
					ageTSize = cv2.getTextSize(age, cv2.FONT_HERSHEY_SIMPLEX,0.5,2)[0]
					ageLoc = (int(t_x + t_w / 2 - (ageTSize[0]) / 2),int(t_y+t_h+15))
					cv2.putText(imgDisplay,gender,genderLoc,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
					cv2.putText(imgDisplay,age,ageLoc,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

			textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
			textX = int(t_x + t_w / 2 - (textSize[0]) / 2)
			if 'gender' in locals():
				textY = int(t_y - genderTSize[1])-6
			else:
				textY = int(t_y)
			textLoc = (textX, textY)

			cv2.rectangle(imgDisplay, (t_x, t_y),
						  (t_x + t_w , t_y + t_h),
						  rectColor ,2)

			cv2.putText(imgDisplay, text, textLoc,
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (255, 255, 255), 2)

		'''
		a = points.shape
		if a[0] > 0:
			# print(points.shape)
			if a[1] > 0:

				for i in range(5):
					# print(points.shape)
					cv2.circle(imgDisplay, (points[i,0],points[i+5,0]), 1, (255, 0, 0), 2)
		'''

	def popup(self,fid,imgtk):

		faceAttr = self.faceAttributesList[str(fid)]

		win = tk.Toplevel()
		win.resizable(width=False,height=False)
		win.wm_title("Register Face")

		frame = tk.ttk.Frame(win,border=2,relief=tk.GROOVE)
		frame.pack(fill=tk.X,padx=5,pady=5)

		imgLabel = tk.Label(frame,image=imgtk,relief=tk.GROOVE)
		imgLabel.imgtk = imgtk
		imgLabel.pack(fill=tk.X)


		nameFrame = tk.ttk.Frame(frame)
		nameFrame.pack(fill=tk.X,pady=5)

		nameLabel = tk.ttk.Label(nameFrame,text='{0:15}\t:'.format('Name'))
		nameLabel.pack(side=tk.LEFT,padx=5)

		nameEntry = tk.ttk.Entry(nameFrame)
		name = faceAttr.awsID
		if faceAttr.awsID in self.faceNamesList.keys():
			name = self.faceNamesList[faceAttr.awsID]
		nameEntry.insert(0, name)
		nameEntry.selection_range(0,tk.END)
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

		perFrame = tk.ttk.Label(frame, text='{0:15}\t: {1}'.format('Blur Per Value', faceAttr.blurPer))
		perFrame.pack(fill=tk.X, padx=5)

		blurExtentFrame = tk.ttk.Label(frame, text='{0:15}\t: {1}'.format('Blur Extent', faceAttr.blurExtent))
		blurExtentFrame.pack(fill=tk.X, padx=5)

		genderFrame = tk.ttk.Label(frame,text='{0:15}\t: {1}'.format('Gender',faceAttr.gender))
		genderFrame.pack(fill=tk.X,padx=5)

		genderConfidenceFrame = tk.ttk.Label(frame,text='{0:15}\t: {1}'.format('Gender Confidence',faceAttr.genderConfidence))
		genderConfidenceFrame.pack(fill=tk.X,padx=5)

		ageRangeFrame = tk.ttk.Label(frame,text='{0:15}\t: {1} - {2}'.format('Age Range',faceAttr.ageRangeLow,faceAttr.ageRangeHigh))
		ageRangeFrame.pack(fill=tk.X,padx=5)

		detectedTimeFrame = tk.ttk.Label(frame, text='{0:15}\t: {1}'.format('Detected Time', faceAttr.detectedTime))
		detectedTimeFrame.pack(fill=tk.X, padx=5)

		recognizedTimeFrame = tk.ttk.Label(frame, text='{0:15}\t: {1}'.format('Recognized Time', faceAttr.recognizedTime))
		recognizedTimeFrame.pack(fill=tk.X, padx=5)

		submit = tk.ttk.Button(frame,text='Submit',width=10,command=lambda:self.submit(faceAttr.awsID,nameEntry.get(),win))
		submit.pack(pady=5)
		win.bind('<Return>',lambda _:self.submit(faceAttr.awsID,nameEntry.get(),win))

		nameEntry.focus()

	def submit(self,awsID,name,win):
		if not awsID == name:
			self.faceNamesList[awsID] = name
			logger.info('Face with ID {} has been set to the name {}'.format(awsID,name))
		win.destroy()

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

	def detectBlur(self,img,thresh=35):
		'''
		Code Implementation from Paper
		[link: https://www.cs.cmu.edu/~htong/pdf/ICME04_tong.pdf]
		'''
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# im = Image.fromarray(color).convert('F')
		Emax1, Emax2, Emax3 = blurDetector.algorithm(gray)
		per, BlurExtent = blurDetector.ruleset(Emax1, Emax2, Emax3, thresh)

		print('BlurExtent: {}\tper: {}'.format(BlurExtent,per))

		return per, BlurExtent

	def detectFace(self,imgDisplay, roi):
		### Avoid use imgDisplay = frame
		frame = imgDisplay.copy()

		self.frame_count += 1
		self.tracker.deleteTrack(imgDisplay)

		points = np.empty((0,0))

		if (self.frame_count%self.frame_interval) == 0:
			# t1 = time.time()
			bounding_boxes,points = align.detect_face.detect_face(roi, minsize, pnet, rnet, onet, threshold, factor,use_thread=self.detectionThread)
			# print(time.time()-t1,'face detect elapsed')
			for (x1, y1, x2, y2, acc) in bounding_boxes:
				### add back cut out region
				x1+=self.ROIx1
				y1+=self.ROIy1
				x2+=self.ROIx1
				y2+=self.ROIy1
				matchedFid = self.tracker.getMatchId(imgDisplay,(x1,y1,x2,y2))

				if matchedFid is None:
					faceAttr = FaceAttribute()
					# datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y')
					faceAttr.detectedTime = str(datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y'))
					self.currentFaceID += 1
					self.num_face += 1
					logger.info('Detected a face number {} in rectangle ({},{}) ({},{})'.format(self.num_face,x1, y1, x2, y2))

					faceImg = cropFace(frame,(x1,y1,x2,y2))

					per, blurExtent = self.detectBlur(faceImg)

					blur = True

					if per >= (self.blurFilter/100):
						blur = False

					faceAttr.blurPer = float(per)
					faceAttr.blurExtent = float(blurExtent)

					roll,pitch,yaw = self.getHeadPoseEstimation(faceImg)

					logger.info('Attributes of detected face number {} are per: {}, blur extent:{}, yaw:{}, pitch:{}'.format(self.num_face,per,blurExtent,yaw,pitch))

					if blur:
						logger.warning('Blur image of face number {} is filtered as it per value ({}) is less than blur min zero filter {}'.format(self.num_face,per,(self.blurFilter/100)))
						saveImage('blur_img',faceImg,self.num_face)
						continue

					saveImage('sharp_img',faceImg,self.num_face)

					if abs(yaw) > self.yawFilter:
						logger.warning('Face number {} is filtered as yaw angle ({}) exceeded yaw filter ({})'.format(self.num_face,abs(yaw),self.yawFilter))
						continue
					elif abs(pitch)>self.pitchFilter:
						logger.warning('Face number {} is filtered as pitch angle ({}) exceeded pitch filter ({})'.format(self.num_face, abs(pitch), self.pitchFilter))
						continue

					faceAttr.roll = float(roll)
					faceAttr.pitch = float(pitch)
					faceAttr.yaw = float(yaw)

					currentFaceID = str(self.currentFaceID)
					self.faceAttributesList[currentFaceID] = faceAttr

					self.tracker.createTrack(imgDisplay,(x1,y1,x2,y2),currentFaceID)
					logger.info('Tracking new face {} in ({},{}), ({},{})'.format(currentFaceID,x1,y1,x2,y2))

					cropface = cropFace(frame,(x1,y1,x2,y2),crop_factor=self.crop_factor,minHeight=80,minWidth=80)

					saveImage(self.saveImgPath,cropface,self.num_face)

					enc = cv2.imencode('.png',cropface)[1].tostring()

					if self.featureOption.get() == 0:
						t2 = threading.Thread(target=self.AWSRekognition,args=([enc,cropface,currentFaceID]))
					elif self.featureOption.get() == 1:
						t2 = threading.Thread(target=self.AWSDetection,args=([enc,cropface,currentFaceID]))
					else:
						t2 = threading.Thread(target=self.ageGenderEstimation,args=([cropface,currentFaceID]))
					t2.start()

					logger.info('Sending face image number {} to AWS for recognition'.format(currentFaceID))

		self.drawTrackedFace(imgDisplay,points.copy())

class FaceAttribute(object):
	def __init__(self):
		self.awsID = None
		self.yaw = None
		self.roll = None
		self.pitch = None
		self.blurPer = None
		self.blurExtent = None
		self.similarity = 'New Face'
		self.gender = None
		self.genderConfidence = None
		self.ageRangeLow = None
		self.ageRangeHigh = None
		self.detectedTime = None
		self.recognizedTime = None

class ImageCard(object):
	def __init__(self):
		self.fid = None
		self.image = None

	def __lt__(self, other):
		if hasattr(other, 'fid'):
			return int(self.fid).__lt__(int(other.fid))

if __name__ == '__main__':
	# NUM_THREADS = 44
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(
			per_process_gpu_memory_fraction=gpu_memory_fraction)
		logger.info('Starting new tensorflow session with gpu memory fraction {}'.format(gpu_memory_fraction))
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
												# intra_op_parallelism_threads=NUM_THREADS,
												log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = align.detect_face.create_mtcnn(
				sess, None)
			
		app = GUI()
		app.showFrame()
		app.mainloop()
