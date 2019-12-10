import dlib
import numpy as np
from database import Database
from config import Config
import datetime

exitedTime = None

class Tracker:
	def __init__(self,*args,**kwargs):
		self.faceTrackers = {}
		self.fidsToDelete = []
		self.trackingQuality = 5
		self.videoFrameSize = np.empty((0,0,0))
		self.outOfScreenThreshold = 0.4
		self.conf = Config('config.ini')
		self.db = Database(self.conf['host'], self.conf['user'], self.conf['password'], self.conf['database'])

	def createTrack(self,imgDisplay,boundingBox,currentFaceID):

		x1,y1,x2,y2 = boundingBox

		w = int(x2-x1)
		h = int(y2-y1)
		x = int(x1)
		y = int(y1)

		# print('tracker'+str(currentFaceID))
		tracker = dlib.correlation_tracker()
		# tracker.start_track(imgDisplay,dlib.rectangle(x-10,y-20,x+w+10,y+h+20))
		tracker.start_track(imgDisplay,dlib.rectangle(x,y,x+w,y+h))
		# tracker.start_track(imgDisplay,dlib.rectangle(x-50,y-50,x+w+50,y+h+50))

		self.faceTrackers[currentFaceID] = tracker

	def deleteAll(self):
		for fid in self.faceTrackers.keys():
			self.fidsToDelete.append(fid)

	def appendDeleteFid(self,fid):
		self.fidsToDelete.append(fid)

	def getRelativeOutScreen(self,fid):
		tracked_position = self.faceTrackers[fid]

		left = tracked_position.get_position().left()
		right = tracked_position.get_position().right()
		top = tracked_position.get_position().top()
		bottom = tracked_position.get_position().bottom()

		width = tracked_position.get_position().width()
		height = tracked_position.get_position().height()

		area = width*height

		outScreenX = 0
		outScreenY = 0
		
		if left < 0:
			outScreenX += 0-left

		if right > self.videoFrameSize[1]:
			outScreenX += right-self.videoFrameSize[1]

		if top < 0:
			outScreenY += 0-top

		if bottom > self.videoFrameSize[0]:
			outScreenY += bottom-self.videoFrameSize[0]

		outScreenIntersect = outScreenX*outScreenY
		outScreenArea = outScreenX*height + outScreenY*width - outScreenIntersect
		relativeOutScreen = outScreenArea/area

		return relativeOutScreen

	def deleteTrack(self,imgDisplay):
		global exitedTime
		for fid in self.faceTrackers.keys():			
			trackingQuality = self.faceTrackers[fid].update(imgDisplay)

			if trackingQuality < self.trackingQuality:
				self.fidsToDelete.append(fid)
				continue

			relativeOutScreen = self.getRelativeOutScreen(fid)

			if relativeOutScreen >= self.outOfScreenThreshold:
				# print("Face Out of Screen")
				self.fidsToDelete.append(fid)
				# face exit time
				exitedTime = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

				# store and update data to the Demographic table
				try:
					stmt = "INSERT INTO FaceExitTime (ExitedTime) VALUES (%s);"
					update_table = "UPDATE Demographic inner join FaceExitTime ON (Demographic.ID = FaceExitTime.ID) SET Demographic.ExitedTime = FaceExitTime.ExitedTime;"
					param = exitedTime
					cursor = self.db.db.cursor()
					cursor.execute(stmt, (param,))
					cursor.execute(update_table)
					print("DB exit time store succesfully at ", exitedTime)
					cursor.close()
					self.db.db.commit()
				except Exception as e:
					print("Error to write FaceExitTime -> ", e)

		while len(self.fidsToDelete) > 0:
			fid = self.fidsToDelete.pop()
			self.faceTrackers.pop(fid,None)

		return exitedTime # important to update table time


	def getMatchId(self,imgDisplay,boundingBox):

		x1,y1,x2,y2 = boundingBox

		w = int(x2-x1)
		h = int(y2-y1)
		x = int(x1)
		y = int(y1)

		##calculate centerpoint
		x_bar = x+0.5*w
		y_bar = y+0.5*h

		matchedFid = None
		for fid in self.faceTrackers.keys():
			tracked_position = self.faceTrackers[fid].get_position()
			
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

				# self.faceTrackers[fid].start_track(imgDisplay,dlib.rectangle(x-10,y-20,x+w+10,y+h+20))
				self.faceTrackers[fid].start_track(imgDisplay,dlib.rectangle(x,y,x+w,y+h))
				# self.faceTrackers[fid].start_track(imgDisplay,dlib.rectangle(x-50,y-50,x+w+50,y+h+50))

		return matchedFid
