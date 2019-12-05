class Face(object):
	def __init__(self):
		self.valid = False
		self.validRecognize = False
		self.awsID = None
		self.faceConfidence = None
		self.yaw = None
		self.roll = None
		self.pitch = None
		self.sharpness = None
		self.similarity = 'New Face'
		self.gender = None
		self.genderConfidence = None
		self.emotion = None
		self.exitedTime = None
		self.ageLow = None
		self.ageHigh = None
		self.detectedTime = None
		self.recognizedTime = None
		self.imgtk = None
		self.location = None

	def __getitem__(self,key):
		return getattr(self,key)

	def __setitem__(self,key,value):
		setattr(self,key,value)
