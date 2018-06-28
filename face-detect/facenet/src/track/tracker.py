import dlib

class Tracker:
	def __init__(self,*args,**kwargs):
		self.faceTrackers = {}
		self.faceID = {}
		self.fidsToDelete = []
		self.trackingQuality = 4

	def createTrack(self,imgDisplay,x,y,w,h,currentFaceID):
		print('Creating new tracker'+str(currentFaceID))
		tracker = dlib.correlation_tracker()
		#tracker.start_track(imgDisplay,dlib.rectangle(x-10,y-20,x+w+10,y+h+20))
		tracker.start_track(imgDisplay,dlib.rectangle(x,y,x+w,y+h))

		self.faceTrackers[currentFaceID] = tracker

	def appendDeleteFid(self,fid):
		self.fidsToDelete.append(fid)

	def deleteTrack(self,imgDisplay):
		for fid in self.faceTrackers.keys():
			trackingQuality = self.faceTrackers[fid].update(imgDisplay)
			if trackingQuality < self.trackingQuality:
				self.fidsToDelete.append(fid)

		while len(self.fidsToDelete) > 0:
			fid = self.fidsToDelete.pop()
			self.faceTrackers.pop(fid,None)


	# def getMatchId(self,x,y,w,h):
	def getMatchId(self,imgDisplay,x,y,w,h):
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

				#self.faceTrackers[fid].start_track(imgDisplay,dlib.rectangle(x-10,y-20,x+w+10,y+h+20))
				self.faceTrackers[fid].start_track(imgDisplay,dlib.rectangle(x,y,x+w,y+h))

		return matchedFid
