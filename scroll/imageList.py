import tkinter as tk
from tkinter import ttk
import sys
import os

# curPath = os.path.dirname(__file__)
# sys.path.insert(0, curPath)
from scroll.face import Face
# from face import Face
import cv2
from PIL import ImageTk,Image

def resizeImage(sizeX,sizeY,img):
	height,width,_ = img.shape
	scaleY = sizeY/height
	scaleX = sizeX/width
	resizeImg = cv2.resize(img,(0,0),fx=scaleX,fy=scaleY)
	return resizeImg

class Popup(tk.Toplevel):
	def __init__(self,master,faceObj):
		tk.Toplevel.__init__(self,master)
		self.master = master
		self.resizable(width=False,height=False)

		self.title("Register Face")

		frame = ttk.Frame(self,border=2,relief=tk.GROOVE)
		frame.pack(fill=tk.X,padx=5,pady=5)

		self.blacklist = []
		# self.faceNamesList = {}

		self.imgLabel = tk.Label(frame,image=faceObj['imgtk'],relief=tk.GROOVE)
		self.imgLabel.imgtk = faceObj['imgtk']
		self.imgLabel.pack(fill=tk.X)   

		nameFrame = ttk.Frame(frame)
		nameFrame.pack(fill=tk.X,pady=5)

		nameLabel = ttk.Label(nameFrame,text='{0:15}\t:'.format('Name'))
		nameLabel.pack(side=tk.LEFT,padx=5)

		nameEntry = ttk.Entry(nameFrame)
		name = 'None'
		if faceObj['awsID'] in self.master.db['faceNameList'].keys():
			name = self.master.db['faceNameList'][faceObj['awsID']]
		nameEntry.insert(0, name)
		nameEntry.selection_range(0,tk.END)
		nameEntry.pack(side=tk.LEFT)

		faceid = ttk.Label(frame,text='{0:15}\t: {1}'.format('Face ID',faceObj['awsID']))
		faceid.pack(fill=tk.X,padx=5)

		similarityFrame = ttk.Label(frame,text='{0:15}\t: {1}'.format('Similarity',faceObj['similarity']))
		similarityFrame.pack(fill=tk.X,padx=5)

		faceConfidenceFrame = ttk.Label(frame,text='{0:15}\t: {1}'.format('Face Confidence',faceObj['faceConfidence']))
		faceConfidenceFrame.pack(fill=tk.X,padx=5)

		rollFrame = ttk.Label(frame,text='{0:15}\t: {1}'.format('Roll(degree)',faceObj['roll']))
		rollFrame.pack(fill=tk.X,padx=5)

		yawFrame = ttk.Label(frame,text='{0:15}\t: {1}'.format('Yaw(degree)',faceObj['yaw']))
		yawFrame.pack(fill=tk.X,padx=5)

		pitchFrame = ttk.Label(frame,text='{0:15}\t: {1}'.format('Pitch(degree)',faceObj['pitch']))
		pitchFrame.pack(fill=tk.X,padx=5)

		blurFrame = ttk.Label(frame, text='{0:15}\t: {1}'.format('Sharpness',faceObj['sharpness']))
		blurFrame.pack(fill=tk.X, padx=5)

		genderFrame = ttk.Label(frame,text='{0:15}\t: {1}'.format('Gender',faceObj['gender']))
		genderFrame.pack(fill=tk.X,padx=5)

		genderConfidenceFrame = ttk.Label(frame,text='{0:15}\t: {1}'.format('Gender Confidence',faceObj['genderConfidence']))
		genderConfidenceFrame.pack(fill=tk.X,padx=5)

		ageRangeFrame = ttk.Label(frame,text='{0:15}\t: {1} - {2}'.format('Age Range',faceObj['ageLow'],faceObj['ageHigh']))
		ageRangeFrame.pack(fill=tk.X,padx=5)

		emotionFrame = ttk.Label(frame, text='{0:15}\t: {1}'.format('Emotion', faceObj['emotion']))
		emotionFrame.pack(fill=tk.X, padx=5)

		detectedTimeFrame = ttk.Label(frame, text='{0:15}\t: {1}'.format('Detected Time',faceObj['detectedTime']))
		detectedTimeFrame.pack(fill=tk.X, padx=5)

		recognizedTimeFrame = ttk.Label(frame, text='{0:15}\t: {1}'.format('Recognized Time',faceObj['recognizedTime']))
		recognizedTimeFrame.pack(fill=tk.X, padx=5)

		locationFrame = ttk.Label(frame, text='{0:15}\t: {1}'.format('Location',faceObj['location']))
		locationFrame.pack(fill=tk.X, padx=5)

		submit = ttk.Button(frame,text='Submit',width=10,command=lambda:self.submit(faceObj['awsID'],nameEntry.get()))
		submit.pack(pady=5)

		# if self.face.awsID not in self.blacklist:
		# 	addtoblacklist = ttk.Button(frame, text='Add to Blacklist',
		# 								   command=lambda: self.destroy() or self.add_to_blacklist(self.face.awsID))
		# 	addtoblacklist.pack(pady=5)
		# else:
		# 	addtoblacklist = ttk.Button(frame, text='Remove from Blacklist',
		# 								   command=lambda: self.destroy() or self.rm_from_blacklist(self.face.awsID))
		# 	addtoblacklist.pack(pady=5)
		
		# self.bind('<Return>',lambda _:self.submit(self.face.awsID,nameEntry.get()))

		nameEntry.focus()
		
	def submit(self,awsID,name):
		if len(name) != 0 and name != 'None':
			self.master.db['faceNameList'][awsID] = name

		self.destroy()

if __name__ =='__main__':
	# not yet complete
	root = tk.Tk()
	faceObj = Face()
	faceObj['awsID'] = 1234
	root.database = {}
	root.database['faceNameList'][faceObj['awsID']] = 'John'

	img = cv2.imread('icon.jpg')
	cv2image = resizeImage(100,100,img)
	cv2image = cv2.cvtColor(cv2image,cv2.COLOR_BGR2RGBA)
	img = Image.fromarray(cv2image)
	faceObj['imgtk'] = ImageTk.PhotoImage(image=img)

	popup = Popup(root,faceObj)
	root.mainloop()