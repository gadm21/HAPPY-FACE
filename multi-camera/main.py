import tkinter as tk
from tkinter import ttk
import cv2
from PIL import ImageTk, Image
import threading
import notebook
from menu import menu

import time
from scroll import scroll, face
from predict import predict
import track.tracker as track
import datetime

from config import Config
from util import Util

import aws.rekognition as aws
from database import Database

class GUI(tk.Tk):
	def __init__(self,*args,**kwargs):
		tk.Tk.__init__(self,*args,**kwargs)

		self.protocol("WM_DELETE_WINDOW", self._quit)
		self.appIcon = tk.Image("photo", file="asset/tapway.png")
		self.wm_iconphoto(True,self.appIcon)
		self.title('Tapway Face System')

		self.conf = Config('config.ini')
		self.db = Database(self.conf['host'],self.conf['user'],self.conf['password'],self.conf['database'])

		self.nb = notebook.Tab(self)
		self.nb.pack(side=tk.LEFT)

		self.imgListframe = scroll.VerticalScrolledFrame(self, width=250, borderwidth=2, relief=tk.GROOVE, background="light gray")
		self.imgListframe.pack(side=tk.RIGHT,fill='y')

		self.cameraJobID = {}
		self.tracker = []

		self.predict = predict.Predict()

		self.camera = []
		self.threads = []

		self.frameCount = []

		self.AWSthreshold = 70
		self.crop_factor = 0.20	

		self.menubar = menu.MenuBar(self)
		self.config(menu=self.menubar)

		self.FIX_HEIGHT = 480 #1280
		self.FIX_WIDTH = 640 #1024

		self.fixSize = True #True

		for i in range(0,self.conf['totalCamera']):
			cap = cv2.VideoCapture(self.conf['cameraSrc'][i])
			### later need handle flag if no flag come in
			_,frame = cap.read()
			height,width,_ = frame.shape
			self.camera.append(cap)

			if self.fixSize:
				self.nb.addTab(self.FIX_HEIGHT,self.FIX_WIDTH)
			else:
				self.nb.addTab(height,width)
			
			self.frameCount.append(0)
			tracker = track.Tracker()
			tracker.videoFrameSize = frame.shape
			self.tracker.append(tracker)

		self.lock = threading.Lock()

	def run(self):
		if self.conf['cameraMultiThread']:
			for i in range(0,self.conf['totalCamera']):
				thread = threading.Thread(target=self.showFrame,args=([i]))
				self.threads.append(thread)
				self.threads[i].start()
		else:
			for i in range(0,self.conf['totalCamera']):
				self.showFrame(i)
		
	def showFrame(self,index):
		flag,frame = self.camera[index].read()
		if flag == True:
			self.tracker[index].deleteTrack(frame)
			if (self.frameCount[index]%self.conf['frameInterval']) == 0:
				self.detectFaceFlow(index,frame)

			if self.fixSize:
				outputImg = cv2.resize(frame,(self.FIX_WIDTH,self.FIX_HEIGHT))
			else:	
				outputImg = frame.copy()

			self.drawTrackedFace(index,outputImg)
			self.nb.display(index,outputImg)
			self.frameCount[index] += 1
		else:
			print('Camera {0} has no frame'.format(index+1))
		self.cameraJobID[index] = self.nb.after(10,lambda :self.showFrame(index))

	def detectFaceFlow(self,index,frame):
		bounding_boxes = self.predict.detectFace(frame)
		for (x1, y1, x2, y2, acc) in bounding_boxes:
			x1 = int(x1)
			y1 = int(y1)
			x2 = int(x2)
			y2 = int(y2)

			matchedFid = self.tracker[index].getMatchId(frame,(x1,y1,x2,y2))

			if matchedFid is None:			
				faceImg = Util.cropFace(frame,(x1,y1,x2,y2))
				faceObj = self.predict.filterFace(faceImg,self.conf.getFilter())

				if faceObj['valid'] is False:
					continue
				faceObj['location'] = 'camera {}'.format(str(index))
				### use lock to deal with multithread
				with self.lock:
					fid = self.db.generateID()
					self.tracker[index].createTrack(frame,(x1,y1,x2,y2),fid)
					self.db.addFace(fid,faceObj)
					faceImg = Util.cropFace(frame,(x1,y1,x2,y2),crop_factor=self.crop_factor,minHeight=80,minWidth=80)

					####### open new thread here cause close window have error
					task = None
					if self.conf['featureOption']==0:
						task = threading.Thread(target=self.demographicFlow,args=([index,fid,faceImg]))
					elif self.conf['featureOption']==1:
						task = threading.Thread(target=self.recognizeFlow,args=([index,fid,faceImg]))
					
					task.start()

	def updateImageList(self,fid,faceImg):
		cv2image = Util.resizeImage(100,100,faceImg)
		cv2image = cv2.cvtColor(cv2image,cv2.COLOR_BGR2RGBA)
		img = Image.fromarray(cv2image)
		self.db['faceList'][fid]['imgtk'] = ImageTk.PhotoImage(image=img)
		self.imgListframe.addFaceToImageList(fid,self.db['faceList'][fid])

	def demographicFlow(self,index,fid,faceImg):
		faceObj = aws.Detection(faceImg)

		# faceObj is None if exception happens in AWSDetection
		if faceObj is None:
			self.tracker[index].appendDeleteFid(fid)
			self.db['faceList'].pop(fid, None)
			return

		# Update the List
		self.db['faceList'][fid]['awsID'] = str(fid)
		self.db['faceList'][fid]['gender'] = faceObj['gender']
		self.db['faceList'][fid]['genderConfidence'] = faceObj['genderConfidence']
		self.db['faceList'][fid]['ageLow'] = faceObj['ageLow']
		self.db['faceList'][fid]['ageHigh'] = faceObj['ageHigh']
		# self.db['faceList'][fid]['blob'] = faceObj['blob']

		self.updateImageList(fid,faceImg)
		self.db.addFaceToDemographic(self.db['faceList'][fid])

	def recognizeFlow(self,index,fid,faceImg):
		faceObj = aws.Recognition(faceImg,self.AWSthreshold,self.db['faceList'][fid]['validRecognize'])

		# faceObj is None if exception happens in AWSRekognition
		if faceObj is None:
			self.tracker[index].appendDeleteFid(fid)
			self.db['faceList'].pop(fid, None)
			return

		# Update the List
		self.db['faceList'][fid]['awsID'] = faceObj['awsID']
		self.db['faceList'][fid]['similarity'] = faceObj['similarity']
		self.db['faceList'][fid]['recognizedTime'] = faceObj['recognizedTime']

		# Note: The face that recognized before does not put to the imageList
		if faceObj['similarity']=='New Face':
			self.updateImageList(fid,faceImg)
		else:
			### only debug purpose - oka
			self.updateImageList(fid,faceImg)

	def drawTrackedFace(self,index,outputImg):

		for fid in self.tracker[index].faceTrackers.keys():
			position = self.tracker[index].faceTrackers[fid].get_position()

			x1 = int(position.left())
			y1 = int(position.top())
			x2 = int(position.right())
			y2 = int(position.bottom())

			rect = (x1,y1,x2,y2)

			if self.db['faceList'][fid] is None:
				"""
				this condition will happen if demographicFlow Thread pop up the faceList,
				but the tracker not yet delete here.
				"""
				continue

			awsID = self.db['faceList'][fid]['awsID']

			if awsID is None:
				text = '...Detecting...'
				rectColor = (0,165,255)			
			elif awsID in self.db['faceNameList'].keys():
				text = self.db['faceNameList'][awsID]
				rectColor = (255,0,0)
			else:
				text = awsID
				rectColor = (0,0,255)				
			Util.drawOverlay(text,rect,rectColor,outputImg)

	def _quit(self):
		try:
			for i in self.cameraJobID.keys():
				self.nb.after_cancel(self.cameraJobID[i])
			for camera in self.camera:
				camera.release()

			for thread in self.threads:
				thread.join()
			
			self.db.close()
			self.destroy()
		except Exception as e:
			print(e)

if __name__ == '__main__':
	app = GUI()
	app.run()
	app.mainloop()  
