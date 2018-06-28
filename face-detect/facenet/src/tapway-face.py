#   import facenet libraires
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import os
import align.detect_face
import dlib

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

import tapway_greet

gpu_memory_fraction = 1.0
minsize = 40
# threshold = [ 0.1, 0.2, 0.0001 ]  # three steps's threshold
threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
factor = 0.709 # scale factor

# Kirthi save video stuff
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_vidname = 'tapway-face-output.avi'

def saveImage(saveImgPath,cropFace,numFace):
	if not os.path.isdir(saveImgPath):
		os.mkdir(saveImgPath)
		print('Create new {} path'.format(saveImgPath))
	cv2.imwrite('{}/face_{}.png'.format(saveImgPath,numFace),cropFace)      

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

	print('Crop face AWS h: {} w: {}'.format(crop_h, crop_w))

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

		self.config = self.readConfigFile()

		self.file = self.config['video']
		self.saveImgPath = self.config['imagePath']
		self.BLUR_THRESH = self.config['blurCap']

		self.frame_interval = 8 # Originally self.frame_interval = 10
		self.frame_count = 0

		self.num_face = 0
		self.currentFaceID = 0

		self.tracker = track.Tracker()
		self.crop_factor = 0.25

		self.winfo_toplevel().resizable(width=False,height=False)
		self.winfo_toplevel().title('Tapway Face System')

		self.videoFrame = tk.Frame(root,width=600,height=600,borderwidth=0)
		self.videoFrame.pack(side=tk.LEFT)
		
		self.videoLabel = tk.Label(self.videoFrame)
		self.videoLabel.pack()
		self.camera = cv2.VideoCapture(self.file)
		self.out = cv2.VideoWriter(output_vidname,fourcc, 30.0, (int(self.camera.get(3)), int(self.camera.get(4))))
		
		self.frame = tkgui.VerticalScrolledFrame(root)
		self.frame.pack(side=tk.RIGHT)

		self.imageList = []
		self.top_faces = []
		self.top_face_id = 0

		### read single frame to setup imageList
		'''
		_,frame = self.camera.read()

		height,_,_ = frame.shape
		row = math.ceil(height/120)

		self.addImageList(row*2)
		'''

		self.addImageList(2)

		self.faceNamesList = {}

	def readConfigFile(self):
		with open('config.json') as configuration:
			conf = json.load(configuration)
		return conf

	def resizeImage(self,sizeX,sizeY,img):
		height,width,_ = img.shape
		scaleY = sizeY/height
		scaleX = sizeX/width
		resizeImg = cv2.resize(img,(0,0),fx=scaleX,fy=scaleY)
		return cv2.cvtColor(resizeImg,cv2.COLOR_BGR2RGBA)

	def addImageList(self,row):
		photo = cv2.imread('gui/icon.jpg')
		cv2image = self.resizeImage(100,100,photo)

		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)

		for i in range(row):
			imgLabel = tk.ttk.Button(self.frame.interior,image=imgtk,state='disable')
			imgLabel.imgtk = imgtk
			self.imageList.append(imgLabel)
		self.gridResetLayout()

	def addFaceToImageList(self,fid,cropface,awsID):
		# Tapway greet!!!
		name = None
		if awsID in self.faceNamesList:
			name = self.faceNamesList[awsID]
		tapway_greet.send_slack(awsID, name)
		
		cv2image = self.resizeImage(100,100,cropface)

		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)

		imgLabel = tk.ttk.Button(self.frame.interior,image=imgtk,command=lambda:self.popup(fid,imgtk,awsID))
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

	def show_frame(self):
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
		self.videoLabel.after(1,self.show_frame)
		#self.out.write(frame) #Kirthi save vid

	def AWSRekognition(self,enc,cropface,fid):
		# Kirthi code for filtering good faces
		# resized = cv2.resize(cropface, (100,100), interpolation = cv2.INTER_CUBIC) 
		# lap = cv2.Laplacian(cropface, cv2.CV_64F).std()
		# detector = dlib.get_frontal_face_detector()
		# gray = cv2.cvtColor(cropface, cv2.COLOR_BGR2GRAY)
		# rects = detector(gray, 1) 

		# # h, w, _ = resized.shape
		# # print('Laplacian size: {} h:{} w: {}'.format(lap,h,w))
		
		# not_blurry = lap >= self.BLUR_THRESH
		# # If front face is detected, len(rects > 0
		# #good_face = len(rects) > 0
		# good_face = True
		# print('Laplacian: ', lap)
		# if (not_blurry and good_face):
		# 	#self.top_faces.append(cropface)
		# 	# fname = '{}.jpg'.format(self.top_face_id)
		# 	# cv2.imwrite('./good_faces/'+fname, cropface)
		# 	# print("")
		# 	# print("==============")
		# 	# print("Good candidate")
		# 	# print("==============")
		# 	# print("")
		# 	self.top_face_id += 1
		# else:
		# 	print('Did not send to AWS because image doest not satisfy quality constraints - Please configure config.json if you would like to change the constraints')
		# 	#return None
		
		try:
			res  = aws.search_faces(enc)
			print(res)
			if len(res['FaceMatches']) == 0:
				res = aws.index_faces(enc)
				awsID = res['FaceRecords'][0]['Face']['FaceId']

				self.tracker.faceID[fid] = awsID
				self.addFaceToImageList(fid,cropface,awsID)

				print('New Face ID:',awsID)

			else:
				awsID = res['FaceMatches'][0]['Face']['FaceId']

				self.tracker.faceID[fid] = awsID
				self.addFaceToImageList(fid,cropface,awsID)

				print('Match Face ID {}'.format(awsID))

		except Exception as e:
			# aws rekognition will have error if does not detect any face
			if type(e).__name__ == 'InvalidParameterException':
				### delete the face, since aws cant detect it as a face
				self.tracker.appendDeleteFid(fid)
			else:
				print(e)
			pass

	def drawTrackedFace(self,imgDisplay):

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


	def popup(self,fid,imgtk,awsID):
		print(awsID)
		win = tk.Toplevel()
		win.wm_title("Register Face")

		imgLabel = tk.Label(win,image=imgtk)
		imgLabel.imgtk = imgtk
		imgLabel.pack(fill=tk.X,expand=True)

		faceid = tk.Label(win,text='Face Id: {0}'.format(awsID))
		faceid.pack(fill=tk.X)

		nameFrame = tk.ttk.Frame(win)
		nameFrame.pack(fill=tk.X,pady=5)

		nameLabel = tk.ttk.Label(nameFrame,text='Name: ')
		nameLabel.pack(side=tk.LEFT,padx=5)

		nameEntry = tk.ttk.Entry(nameFrame)
		nameEntry.pack(side=tk.RIGHT,padx=5)

		submit = tk.ttk.Button(win,text='Submit',width=10,command=lambda:self.submit(awsID,nameEntry.get(),win))
		submit.pack(pady=5)

	def submit(self,awsID,name,win):
		self.faceNamesList[awsID] = name
		win.destroy();

	def detectFace(self,imgDisplay):
		### Avoid use imgDisplay = frame
		frame = imgDisplay.copy()
		
		self.frame_count += 1
		self.tracker.deleteTrack(imgDisplay)

		if (self.frame_count%self.frame_interval) == 0:
			bounding_boxes, points = align.detect_face.detect_face(imgDisplay, minsize, pnet, rnet, onet, threshold, factor)
			
			# Kirthi code for detection of facial landmarks using facenet
			# if ((10, 1) == points.shape):
			# 	for i in range(len(points))[:5]:
			# 		x = points[i][0]
			# 		y = points[i+5][0]
			# 		print("({}, {}),".format(x,y))
			# 		#cv2.circle(imgDisplay, (x, y), 3, (0,0,255), -1)
			# 	cv2.imwrite('detect.jpg', imgDisplay)

			# Kirthi code
			# bbox_idx = 0
			# End of Kirthi code
			
			for (x1, y1, x2, y2, acc) in bounding_boxes:

				w = int(x2-x1)
				h = int(y2-y1)
				x = int(x1)
				y = int(y1)

				# cropped = imgDisplay[y:y+h, x:x+w]
				# lap = cv2.Laplacian(cropped, cv2.CV_64F).std()
				# if lap < 20:
				# 	continue
				# gray = cv2.cvtColor(imgDisplay, cv2.COLOR_BGR2GRAY)
				# gx, gy = np.gradient(cropped)
				# gradient = gx + gy
				# blur = gradient.var()
				cv2.imwrite('filter_data/{}.png'.format(self.frame_count), imgDisplay[y:y+h, x:x+w])
			
				# Kirthi code
				# cropped = imgDisplay[y:y+h, x:x+w]
				# cv2.imwrite('./frames/{}_{}.jpg'.format(self.frame_count, bbox_idx), cropped)
				# bbox_idx += 1
				# End of Kirthi code

				matchedFid = self.tracker.getMatchId(imgDisplay,x,y,w,h)

				if matchedFid is None:
					self.currentFaceID += 1
					self.num_face += 1

					self.tracker.createTrack(imgDisplay,x,y,w,h,self.currentFaceID)

					cropface = cropFaceAWS(frame,(x1,x2,y1,y2),self.crop_factor)
					saveImage(self.saveImgPath,cropface,self.num_face)
						
					enc = cv2.imencode('.png',cropface)[1].tostring()
					t2 = threading.Thread(target=self.AWSRekognition,args=([enc,cropface,self.currentFaceID]))
					t2.start()
		
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
