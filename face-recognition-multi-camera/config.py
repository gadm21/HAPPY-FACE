from configparser import ConfigParser

class Config:
	def __init__(self,file):
		config = ConfigParser()
		config.read(file)
		
		self.cameraSrc = []

		self.host = config.get('mysql','host')
		self.user = config.get('mysql','user')
		self.password = config.get('mysql','password')
		self.database = config.get('mysql','database')

		self.saveImgPath = config.get('default','imagePath')
		self.frameInterval = config.getint('default','frameInterval')
		
		self.pitchFilter = config.getfloat('default','pitchFilter')
		self.yawFilter = config.getfloat('default','yawFilter')
		self.blurFilter = config.getfloat('default','blurFilter')
		self.pitchFilterID = config.getfloat('default','pitchFilterID')
		self.yawFilterID = config.getfloat('default','yawFilterID')
		self.blurFilterID = config.getfloat('default','blurFilterID')
		# self.detectionThread = config.getboolean('default','detectionThread')
		
		self.featureOption = config.getint('default','featureOption')

		self.cameraMultiThread = config.getboolean('default','cameraMultiThread')

		self.totalCamera = config.getint('default','totalCamera')

		for i in range(0,self.totalCamera):
			camera = 'cam'+str(i)
			cameraSrc = config.get(camera,'video')
			if cameraSrc.isdigit():
				cameraSrc = int(cameraSrc)
			self.cameraSrc.append(cameraSrc)
	
	def __getitem__(self,key):
		return getattr(self,key)

	def __setitem__(self,key,value):
		setattr(self,key,value)

	def getFilter(self):
		return {
			'blurFilter':self.blurFilter, 'yawFilter': self.yawFilter, 'pitchFilter': self.pitchFilter,
			'blurFilterID':self.blurFilterID, 'yawFilterID': self.yawFilterID, 'pitchFilterID': self.pitchFilterID
		}

	def load(self,file):
		self.cameraSrc = []
		self.saveImgPath = config.get('default','imagePath')
		self.frameInterval = config.getint('default','frameInterval')
		self.pitchFilter.set(config.getfloat('default','pitchFilter'))
		self.yawFilter.set(config.getfloat('default','yawFilter'))
		self.blurFilter.set(config.getfloat('default','blurFilter'))
		self.pitchFilterID.set(config.getfloat('default','pitchFilterID'))
		self.yawFilterID.set(config.getfloat('default','yawFilterID'))
		self.blurFilterID.set(config.getfloat('default','blurFilterID'))
		self.detectionThread = config.getboolean('default','detectionThread')

	def saveChanges(self,file):
		#Update the config.ini file
		print('save')
	
if __name__ == '__main__':
	config = Config('config.ini')
	print(config['cameraSrc'])
